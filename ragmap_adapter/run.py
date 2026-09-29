"""Run STaR's RGB-D 3-D primitive builder behind RAGMAP's object-mapping contract.

    ragmap-run --input /input --output /output [key=value ...]

**What runs.** ``star.scenegraph.scenegraph_constructor_depth.run_scenegraph_generation``,
the released RGB-D counterpart of the LiDAR constructor behind the paper's
memory, unmodified: Tag2Text tags, GroundingDINO boxes (0.25/0.25, NMS 0.5),
TAP masks and captions, SBERT (all-MiniLM-L6-v2) caption features, background
filtering against its fixed captions (``bg_rate`` 0.60), and incremental
association at ``sim_threshold`` 0.7 with 1.0 IoU3D + 0.2 TF-IDF + 0.4 SBERT
(``aggregate_similarities``). Its thresholds are the released
``configs/scenegraph/collection_docker_coda.yaml`` values, unchanged.

**What the adapter supplies** -- upstream released no RGB-D driver, so the part
a driver would do is here, and nothing else:

- *The frame stream.* Upstream's ROS node (``scripts/run_data_collection_lidar.py``,
  ``ObservationHub``) groups frames into 1 s windows with its own
  ``JitterBuffer(desired_timewindow=1.0)`` and the scene-graph worker keeps the
  last frame of each (``get_observation_by_window``). The adapter replays the
  exported frames through that same ``JitterBuffer`` class, by timestamp, and
  feeds the last frame of every emitted window to
  ``run_scenegraph_generation`` through its own ``iterator`` parameter. As in
  the live node, the trailing partial window is never emitted.
- *The camera keys the RGB-D constructor reads and the released config lacks*
  (``rgb_camera_matrix``, ``depth_camera_matrix``, ``depth_to_rgb_matrix``,
  ``projection_depth``): our intrinsics for both cameras, identity extrinsics
  (the contract's depth is registered to the RGB frame), no debug projection.
- *Paths*: weights under ``/weights``, results under ``/output``.

**Where it departs from upstream**, each counted in ``run.json``:

- ``gobs_to_detection_list_depth`` returns two values instead of five when a
  frame has no masks, so upstream's unpacking raises; its loop's
  ``except Exception`` then ends the whole run silently and saves a truncated
  map. The adapter wraps that one function to return the five-tuple upstream's
  own "no detections" branch expects (``frames_without_masks``). Replaces a
  crash only.
- Any other exception inside upstream's loop is caught and printed by upstream,
  which then saves what it had. The adapter checks that every selected frame
  was consumed and fails the run otherwise, rather than publishing a silently
  truncated map.
- ``vis_interval`` (a debug Open3D viewer, every 10000 frames upstream) is set
  out of reach: there is no display in the container.

**OUTPUT** (``ragmap.objects.base``): ``objects.jsonl``, one object per STaR
map object (background objects are upstream's own separate list and are not
exported), ``pointclouds/<id>.ply``, ``run.json``, plus upstream's own
``pcd/full_pcd.pkl.gz`` and its annotated keyframes under ``annotated_rgb/``.
STaR's objects carry TAP captions but no class label (its final
``class_methods`` pass is configured but never called), so ``label`` is the
first TAP caption and ``caption`` the accumulated ones.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

STAR_ROOT = Path(os.environ.get("STAR_ROOT", Path(__file__).resolve().parents[1]))
UPSTREAM_CONFIG = STAR_ROOT / "configs" / "scenegraph" / "collection_docker_coda.yaml"
GD_CONFIG = Path(os.environ.get(
    "STAR_GD_CONFIG",
    "/opt/third_parties/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
))
WEIGHTS = Path(os.environ.get("STAR_WEIGHTS", "/weights"))
SCHEMA = "ragmap.star_primitives/v1"
#: `ObservationHub.__init__`: `JitterBuffer(desired_timewindow=1.0)`.
WINDOW_SECONDS = 1.0
WEIGHT_FILES = {
    "tag2text_path": "tag2text_swin_14m.pth",
    "gd_weights": "groundingdino_swint_ogc.pth",
    "tap_path": "tap_vit_l_v1_0.pkl",
    "tap_merge_path": "merged_2560.pkl",
}


@dataclass(frozen=True)
class Frame:
    frame_index: int
    rgb: Path
    depth: Path
    pose: np.ndarray
    timestamp: float


def _git_sha() -> str:
    sha = os.environ.get("STAR_GIT_SHA", "").strip()
    if sha and sha != "unknown":
        return sha
    try:
        return subprocess.run(
            ["git", "-C", str(STAR_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def read_input(input_dir: Path) -> tuple[dict[str, Any], list[Frame]]:
    meta = json.loads((input_dir / "meta.json").read_text(encoding="utf-8"))
    if meta.get("up_axis") != "z":
        raise ValueError(f"meta.json up_axis is {meta.get('up_axis')!r}; this adapter expects 'z'.")
    frames: list[Frame] = []
    for line_no, raw in enumerate((input_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        record = json.loads(raw)
        if record.get("timestamp") is None:
            raise ValueError(
                f"frames.jsonl:{line_no} has no timestamp; STaR windows its frames by time (1 s)."
            )
        frames.append(Frame(
            frame_index=int(record["frame_index"]),
            rgb=input_dir / record["rgb"],
            depth=input_dir / record["depth"],
            pose=np.asarray(record["pose"], dtype=np.float64).reshape(4, 4),
            timestamp=float(record["timestamp"]),
        ))
    if not frames:
        raise ValueError(f"{input_dir / 'frames.jsonl'} holds no frames.")
    frames.sort(key=lambda frame: frame.timestamp)
    return meta, frames


def select_frames(frames: Sequence[Frame]) -> tuple[list[Frame], int]:
    """The last frame of each 1 s window, exactly as upstream's live node forms them.

    Returns the selected frames and how many frames sat in the trailing
    window, which upstream's node never emits either.
    """

    from star.utils.util import JitterBuffer

    buffer = JitterBuffer(desired_timewindow=WINDOW_SECONDS)
    selected: list[Frame] = []
    for frame in frames:
        window = buffer.add(frame, frame.timestamp)
        if window:
            selected.append(window[-1])
    trailing = len(buffer.close() or [])
    return selected, trailing


def build_config(meta: dict[str, Any], output_dir: Path, overrides: Sequence[str]) -> Any:
    from omegaconf import OmegaConf

    upstream = OmegaConf.load(UPSTREAM_CONFIG)
    # The released file is a Hydra group member: `sequence: ${..sequence}`
    # resolves against its parent, so it is nested the same way here.
    root = OmegaConf.create({"sequence": "ragmap", "vis_sequence": "ragmap", "scenegraph": upstream})
    OmegaConf.resolve(root)
    cfg = root.scenegraph
    intrinsics = [
        [float(meta["fx"]), 0.0, float(meta["cx"])],
        [0.0, float(meta["fy"]), float(meta["cy"])],
        [0.0, 0.0, 1.0],
    ]
    out = output_dir.as_posix().rstrip("/")
    supplied = {
        # The keys the RGB-D constructor reads and the released config lacks.
        "rgb_camera_matrix": intrinsics,
        "depth_camera_matrix": intrinsics,
        "depth_to_rgb_matrix": np.eye(4).tolist(),
        "projection_depth": False,
        # Weights and outputs.
        **{key: str(WEIGHTS / name) for key, name in WEIGHT_FILES.items()},
        "gd_path": str(GD_CONFIG),
        "sbert_path": str(WEIGHTS / "all-MiniLM-L6-v2"),
        "save_vis_path": f"{out}/vis/",
        "save_vis_proj_path": f"{out}/project/",
        "save_pcd_path": f"{out}/pcd/",
        "annotated_rgb_path": f"{out}/annotated_rgb/",
        # A debug Open3D viewer; there is no display here.
        "vis_interval": 10**9,
    }
    for key, value in supplied.items():
        OmegaConf.update(cfg, key, value, force_add=True)
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Override {item!r} is not key=value.")
        OmegaConf.update(cfg, key.strip(), OmegaConf.from_dotlist([f"v={value}"]).v, force_add=True)
    for key, name in WEIGHT_FILES.items():
        if not Path(cfg[key]).is_file():
            raise FileNotFoundError(f"{key}: {cfg[key]} does not exist (mount the STaR weights at {WEIGHTS}).")
    return cfg


class FrameSource:
    """A finite iterator over the selected frames, in upstream's tuple shape.

    ``iter_by_event_depth`` blocks on a live queue forever; this is passed as
    ``run_scenegraph_generation``'s ``iterator`` instead, so the loop ends by
    ``StopIteration``. ``exhausted`` is how the adapter tells that normal end
    from upstream's ``except Exception`` ending it early.
    """

    def __init__(self, frames: Sequence[Frame], depth_scale: float) -> None:
        self.frames = list(frames)
        self.depth_scale = float(depth_scale)
        self.consumed = 0
        self.exhausted = False

    def __call__(self, _data_source: Any, _loop_event: Any) -> Iterator[tuple[Any, ...]]:
        return self._iterate()

    def _iterate(self) -> Iterator[tuple[Any, ...]]:
        from PIL import Image

        for frame in self.frames:
            with Image.open(frame.rgb) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
            with Image.open(frame.depth) as image:
                depth = np.asarray(image, dtype=np.float32) / self.depth_scale
            if depth.shape != rgb.shape[:2]:
                raise ValueError(f"Frame {frame.frame_index}: depth {depth.shape} is not RGB {rgb.shape[:2]}.")
            self.consumed += 1
            yield (frame.frame_index, rgb, depth, frame.pose.copy(), frame.timestamp)
        self.exhausted = True


def install_patches(sgd: Any, counters: dict[str, int]) -> None:
    """The one crash replacement; see the module docstring."""

    from star.some_class.map_class import DetectionList

    original = sgd.gobs_to_detection_list_depth

    def gobs_to_detection_list_depth(*args: Any, **kwargs: Any) -> Any:
        gobs = kwargs.get("gobs")
        if gobs is not None and len(gobs) == 0:
            counters["frames_without_masks"] += 1
            return DetectionList(), DetectionList(), None, {}, []
        return original(*args, **kwargs)

    sgd.gobs_to_detection_list_depth = gobs_to_detection_list_depth


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if len(colors) == len(points):
        cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), cloud)


def export_objects(pkl: Path, selected: Sequence[Frame], output_dir: Path) -> list[dict[str, Any]]:
    with gzip.open(pkl, "rb") as handle:
        results = pickle.load(handle)
    rows: list[dict[str, Any]] = []
    for position, obj in enumerate(results["objects"]):
        object_id = f"star_{position:04d}"
        points = np.asarray(obj.get("pcd_np", np.zeros((0, 3))), dtype=np.float64).reshape(-1, 3)
        colors = np.asarray(obj.get("pcd_color_np", np.zeros((0, 3))), dtype=np.float64).reshape(-1, 3)
        caption = str(obj.get("caption") or "").strip()
        captions = [part.strip() for part in caption.split(", ") if part.strip()]
        image_idx = [int(index) for index in obj.get("image_idx") or ()]
        frame_indices = sorted({selected[index].frame_index for index in image_idx if 0 <= index < len(selected)})
        cloud = None
        if len(points):
            cloud = f"pointclouds/{object_id}.ply"
            _write_ply(output_dir / cloud, points, colors)
        feature = obj.get("ft")
        rows.append({
            "id": object_id,
            "label": captions[0] if captions else "object",
            "caption": caption or None,
            "crop": None,
            "centroid": points.mean(axis=0).tolist() if len(points) else None,
            "bbox_min": points.min(axis=0).tolist() if len(points) else None,
            "bbox_max": points.max(axis=0).tolist() if len(points) else None,
            "pointcloud": cloud,
            "frame_indices": frame_indices,
            "floor": None,
            "room": None,
            "extra": {
                "captions": captions,
                "num_detections": int(obj.get("num_detections") or 0),
                "n_points": [int(value) for value in obj.get("n_points") or ()],
                "star_image_idx": image_idx,
                "bbox_type": obj.get("bbox_type"),
                # SBERT all-MiniLM-L6-v2 caption feature, detection-weighted
                # mean (`merge_obj2_into_obj1`): what STaR's query side scores.
                "sbert_ft": np.asarray(feature, dtype=np.float32).reshape(-1).tolist() if feature is not None else None,
            },
        })
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragmap-run", description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("overrides", nargs="*", help="key=value overrides of the scenegraph config")
    args = parser.parse_args(argv)
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    run: dict[str, Any] = {"schema": SCHEMA, "status": "running", "star_git_sha": _git_sha(), "overrides": list(args.overrides)}
    timings: dict[str, float] = {}
    started = time.time()
    try:
        sys.path.insert(0, str(STAR_ROOT / "src"))
        meta, frames = read_input(args.input)
        selected, trailing = select_frames(frames)
        if not selected:
            raise ValueError(f"No 1 s window closed over {len(frames)} frames; nothing to map.")
        cfg = build_config(meta, output, args.overrides)
        loaded = time.time()
        import star.scenegraph.scenegraph_constructor_depth as sgd

        timings["import_seconds"] = time.time() - loaded
        counters = {"frames_without_masks": 0}
        install_patches(sgd, counters)
        source = FrameSource(selected, float(meta.get("depth_scale", 1000.0)))
        mapped = time.time()
        sgd.run_scenegraph_generation(cfg, None, None, None, iterator=source)
        timings["scenegraph_seconds"] = time.time() - mapped
        run["counts"] = {
            "frames_input": len(frames),
            "windows": len(selected),
            "frames_in_trailing_window": trailing,
            "frames_processed": source.consumed,
            "frames_without_masks": counters["frames_without_masks"],
        }
        if not source.exhausted:
            raise RuntimeError(
                f"STaR's scene-graph loop stopped after {source.consumed} of {len(selected)} frames: "
                "upstream caught an exception (its traceback is printed above) and saved a truncated map."
            )
        pkl = Path(str(cfg.save_pcd_path)) / "full_pcd.pkl.gz"
        rows = export_objects(pkl, selected, output)
        with (output / "objects.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        run["counts"]["objects"] = len(rows)
        run["config"] = {
            key: cfg[key] for key in (
                "bg_rate", "min_points_threshold", "dbscan_eps", "dbscan_min_points", "voxel_size",
                "max_depth", "filter_dis", "bbox_mode", "use_bg", "use_dam", "spatial_weight",
                "caption_weight", "ft_weight", "sim_threshold",
            )
        }
        run["adapter"] = {
            "window_seconds": WINDOW_SECONDS,
            "frame_per_window": "last",
            "depth_to_rgb_matrix": "identity",
            "rgb_and_depth_camera_matrix": "meta.json intrinsics",
            "vis_interval": cfg.vis_interval,
        }
        run["status"] = "ok"
        return 0
    except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised as the exit code
        run["status"] = "failed"
        run["error"] = f"{type(exc).__name__}: {exc}"
        run["traceback"] = traceback.format_exc()
        traceback.print_exc()
        return 1
    finally:
        timings["total_seconds"] = time.time() - started
        run["timings"] = timings
        (output / "run.json").write_text(json.dumps(run, indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
