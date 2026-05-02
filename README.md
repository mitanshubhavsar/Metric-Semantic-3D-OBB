# Metric-Semantic 3D Reconstruction — CP260 Final Project

**Task:** Given 16 posed images of a desktop PC tower, estimate 3D Oriented Bounding Boxes (OBBs) for back-panel ports: `vga_socket`, `ethernet_socket`, `power_socket`.

**Evaluation:** Polygonal IoU of the projected OBB onto held-out test images.

---

## Repository Structure

```
robotics_proj_2/
├── src/
│   ├── config.py          # Paths, intrinsics, per-entity priors & physical sizes
│   ├── utils.py           # Camera geometry, DLT helpers, OBB visualisation
│   ├── triangulate.py     # Multi-view DLT triangulation
│   ├── sam_segment.py     # SAM segmentation via HuggingFace (point/box prompt)
│   ├── crop_refine.py     # Tight crop → SAM → mask back-projected to full image
│   ├── obb_fit.py         # OBB fitting (PCA + Open3D AABB/OBB)
│   └── pipeline.py        # End-to-end pipeline — outputs answers.json
├── Data/                  # 16 PNGs + poses.json + intrinsic.json
├── outputs/
│   ├── answers.json       # Final submission: 3 OBBs
│   └── validation/        # Projected OBB overlays on all 16 frames
├── docs/
│   └── report.pdf
├── requirements.txt
└── README.md
```

---

## Method Overview

### 1. Center Estimation
3D connector centres are estimated via **multi-view visual triangulation** using manually-verified pixel correspondences (see `triangulate_visual.py` in the reference project). Reprojection errors of 30–60 px are achieved across all 16 frames.

As a complementary approach, `crop_refine.py` implements automated SAM-based centre detection:
1. Project the prior centre into each frame to get a prompt point.
2. Crop a 440×440 window around the projection.
3. Run SAM (ViT-B) with a point prompt at the crop centre.
4. Filter masks by score ≥ 0.65, centroid deviation ≤ 120 px, mask-fraction ≤ 0.50.
5. Back-project accepted mask centroids to world space via DLT.

The visual triangulation prior is used as the final centre for all three entities since it achieves lower reprojection error than the automated SAM-DLT result.

### 2. Rotation Estimation
All back-panel ports share the same face orientation. The rotation matrix is taken directly from the provided VGA socket ground-truth, which aligns:
- **col 0** → panel outward normal (≈ world Z)
- **col 1** → horizontal along panel (≈ world X)
- **col 2** → vertical along panel (≈ world Y)

### 3. Extent Estimation
Half-extents are calibrated against the VGA ground truth:
- **Depth axis (col 0):** 35.4 mm half — same as VGA GT; reflects DLT depth uncertainty rather than physical connector depth.
- **Horizontal/vertical axes:** Scaled from physical connector dimensions using the calibration ratio derived from VGA GT vs physical VGA D-Sub 15 body (≈ 0.77× for horizontal, ≈ 0.97× for vertical).

| Entity | Horiz half (mm) | Vert half (mm) | Depth half (mm) |
|--------|-----------------|----------------|-----------------|
| vga_socket | 11.8 (GT) | 6.1 (GT) | 35.4 (GT) |
| ethernet_socket | 6.1 | 6.6 | 35.4 |
| power_socket | 11.1 | 9.8 | 35.4 |

---

## Running the Pipeline

### Environment Setup

```bash
conda create -n robotics_proj python=3.10
conda activate robotics_proj
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Run Full Pipeline

```bash
cd robotics_proj_2
python src/pipeline.py                          # all entities
python src/pipeline.py --entities ethernet_socket power_socket
```

Outputs: `outputs/answers.json`

### Visualise Results

```python
import sys; sys.path.insert(0, 'src')
import config, utils, cv2, numpy as np, json

poses = utils.load_poses()
K = utils.build_K()
with open('outputs/answers.json') as f:
    answers = json.load(f)

img = utils.load_image_bgr(468)
c2w = poses[468]
for entry in answers:
    img = utils.draw_obb_on_image(img, entry['obb'], c2w, K,
                                   label=entry['entity'][:3],
                                   color=(0, 255, 0))
cv2.imwrite('check.jpg', cv2.resize(img, (1280, 720)))
```

---

## Output Format

`outputs/answers.json` — list of OBB records:

```json
[
  {
    "entity": "vga_socket",
    "obb": {
      "center":   [x, y, z],          // world-space 3D centre (metres)
      "extent":   [e0, e1, e2],       // half-extents along each OBB axis (metres)
      "rotation": [[r00,r01,r02],     // 3×3 rotation matrix; columns = OBB axes
                   [r10,r11,r12],
                   [r20,r21,r22]]
    }
  },
  ...
]
```

---

## Key Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.5.1+cu121 | SAM inference on GPU |
| transformers | ≥ 4.40 | SAM ViT-B via HuggingFace |
| open3d | 0.19.0 | OBB fitting |
| opencv-python | ≥ 4.8 | Image I/O, projection drawing |
| scikit-learn | latest | PCA-based OBB fitting |
| scipy | latest | Outlier filtering |
| numpy | latest | All linear algebra |

---

## Hardware

- 2× NVIDIA RTX 4090 (24 GB) — CUDA 12.2, driver 535.x
- SAM inference: ~2 s/frame on GPU
