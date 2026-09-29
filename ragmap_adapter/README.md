# RAGMAP adapter

`ghcr.io/velythyl/star` runs STaR's RGB-D 3-D primitive builder
(`src/star/scenegraph/scenegraph_constructor_depth.py`, unmodified) behind
RAGMAP's object-mapping container contract (`ragmap/objects/base.py` in
RAGMAP). Upstream code is untouched; everything added is in `ragmap_adapter/`,
`Dockerfile.ragmap` and `.github/workflows/ghcr.yml`. `run.py`'s docstring lists
exactly what the adapter supplies and where it departs from upstream.

    podman run --rm --device nvidia.com/gpu=all \
      -v <input>:/input:ro -v <output>:/output -v <weights>:/weights \
      ghcr.io/velythyl/star:<tag> --input /input --output /output [key=value ...]

`key=value` overrides the scene-graph config (`configs/scenegraph/collection_docker_coda.yaml`).

## Weights (`/weights`)

Upstream's `docker/download_weights.sh` files, plus the two Hugging Face models
the code loads by name, in an HF cache (`HF_HOME=/weights/hf`, offline):

    tag2text_swin_14m.pth  groundingdino_swint_ogc.pth  tap_vit_l_v1_0.pkl  merged_2560.pkl
    hf/  (bert-base-uncased, sentence-transformers/all-MiniLM-L6-v2)

## Input and output

INPUT `/input`: `meta.json` (`fx fy cx cy width height depth_scale up_axis=z`),
`frames.jsonl` (`frame_index observation_id rgb depth pose timestamp`; pose is a
row-major 4x4 OpenCV camera-to-world in metres), `rgb/`, `depth/` (uint16 PNG).

OUTPUT `/output`: `objects.jsonl` (`id label caption centroid bbox_min bbox_max
pointcloud frame_indices extra`), `pointclouds/*.ply`, `run.json`, upstream's
`pcd/full_pcd.pkl.gz` and annotated keyframes in `annotated_rgb/`.
