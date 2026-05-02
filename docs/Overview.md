# CP260 Final Project — 3D OBB Estimation for PC Back-Panel Connectors

> **Authors:** Mitanshu Bhavsar & Ipsita Basak | **Due:** 3 May 2026

Given 16 posed RGB images of a desktop PC tower, we estimate **3D Oriented Bounding Boxes (OBBs)** for five rear-panel connectors: VGA, Ethernet (RJ-45), Power (IEC C14), USB 3.0, and 3.5mm audio jack.

---

## What We Built

A six-module pipeline that:
1. Crops tight windows around each connector and runs SAM to get precise masks
2. Triangulates 3D centers via DLT across multiple views
3. Calibrates extents from known physical connector sizes against the VGA ground truth
4. Shares the VGA rotation across all ports (they all sit on the same flat panel)
5. Writes `outputs/answers.json` and saves visual validation overlays

---

## Quick Start

```bash
# Setup — CUDA 12.1-compatible PyTorch for RTX 4090
conda create -n robotics_proj python=3.10 -y && conda activate robotics_proj
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Run the pipeline (~3–5 min on GPU)
cd robotics_proj_2
python src/pipeline.py                                  # all entities
python src/pipeline.py --no-sam                         # skip SAM, priors only
python src/pipeline.py --entities power_socket ethernet_socket
```

Validation images land in `outputs/validation/`.

---

## Data Layout

```
robotics_proj_2/
├── Data/
│   ├── frame_000319.png … frame_000531.png   (16 frames, 2560×1440)
│   └── poses.json                             (704 camera-to-world 4×4 matrices)
├── intrinsic.json                             (fx=1477, fy=1480, cx=1298, cy=687)
├── sample_answers.json                        (VGA ground truth — scale reference)
├── src/
│   ├── config.py          ← all constants, priors, calibrated extents
│   ├── utils.py           ← camera math, projection, OBB helpers
│   ├── triangulate.py     ← DLT multi-view triangulation (SVD)
│   ├── sam_segment.py     ← SAM ViT-B wrapper (HuggingFace)
│   ├── crop_refine.py     ← crop-then-SAM + mask quality filtering
│   ├── obb_fit.py         ← Open3D / PCA OBB fitting
│   └── pipeline.py        ← orchestrates everything → answers.json
└── outputs/
    ├── answers.json
    └── validation/
```

---

## How the Pipeline Works

### Step 1 — Prior Centers
Each connector has a hand-verified 3D center prior in `config.py`. VGA uses the GT directly. Ethernet and power were triangulated offline by manually annotating pixel correspondences across frames. USB and audio were back-projected from frame 468 at known depth.

### Step 2 — Crop-SAM Segmentation
Full-image SAM fails on ~30px connectors — it grabs the entire I/O panel. The fix: crop a 440×440px window around the projected prior center, run SAM on the crop, then map the mask centroid back to full-image coordinates.

Masks are rejected if:
- SAM confidence < 0.65
- Mask area < 30px²
- Mask fills > 50% of the crop (grabbed surrounding panel, not the connector)
- Centroid drifted > 120px from the projected prior

### Step 3 — Triangulation
Accepted mask centroids from all visible frames feed into DLT triangulation via SVD. Falls back to the prior center if reprojection error > 80px or the result is > 25cm away from the prior.

### Step 4 — Extent Calibration
SAM mask sizes drift frame-to-frame (adjacent connectors blur the mask). Instead, we calibrate against the VGA GT:

> VGA GT measures ≈ 77% of its physical connector face size

That ratio is applied to known physical dimensions of each connector. The **depth axis** is fixed at 35.4mm for all — this matches VGA GT and reflects DLT triangulation smear along the panel normal, not actual connector depth.

### Step 5 — Rotation
All connectors sit on the same rear panel, so they share the VGA GT rotation matrix. No PCA estimation needed.

---

## Final OBBs (`outputs/answers.json`)

| Entity | Center (m) | Horiz half-ext | Vert half-ext |
|--------|-----------|----------------|---------------|
| `vga_socket` | [0.2705, 0.2261, 0.8349] | 11.82 mm | 6.13 mm |
| `ethernet_socket` | [0.2927, 0.2160, 0.7562] | 6.05 mm | 6.59 mm |
| `power_socket` | [0.2911, 0.2150, 0.5261] | 11.11 mm | 9.76 mm |
| `usb_socket` | [0.2700, 0.1999, 0.7738] | 5.59 mm | 11.71 mm |
| `audio_socket` | [0.2669, 0.1932, 0.7335] | 3.45 mm | 4.39 mm |

All share the VGA GT rotation matrix, orthonormal to machine precision ($\|R^\top R - I\|_\infty < 3 \times 10^{-16}$).

The Z-coordinates tell a physically sensible story:
```
Z = 0.835m → VGA        (top of I/O shield)
Z = 0.774m → USB        ↕ 61mm
Z = 0.756m → Ethernet   ↕ 79mm
Z = 0.734m → Audio      ↕ 101mm
Z = 0.526m → Power      ↕ 309mm  (PSU inlet, bottom of tower)
```

---

## Tricky Bits We Hit

**CUDA mismatch** — PyTorch was installed with `cu130` but the driver only supports CUDA 12.2. Reinstalling with `cu121` wheels dropped SAM inference from 45s/frame to ~2s/frame.

**SAM scale problem** — On full 2560×1440 images, SAM reliably grabbed the entire I/O panel instead of individual connectors. Crop-then-SAM fixed it completely.

**Unreliable extent estimation from SAM** — Mask sizes varied wildly between frames. Physical calibration from known connector dimensions proved far more accurate and consistent.

**DLT depth axis confusion** — The largest VGA GT extent axis (35.4mm) is *not* the connector's physical depth. It's triangulation smear along the panel normal from limited camera baseline. Once we understood this, the axis ordering made sense.

---

## Validation

Open any image in `outputs/validation/` to see OBB wireframes sitting tightly on each connector. Frame 468 gives the cleanest back-panel view.

Reprojection of centers in frame 468:

| Entity | Pixel (u, v) | Lands on |
|--------|-------------|----------|
| VGA | (1599, 390) | VGA port centre |
| Ethernet | (1578, 564) | RJ-45 body |
| Power | (1600, 1075) | IEC C14 centre |
| USB | (1615, 545) | USB stack centre |
| Audio | (1609, 645) | 3.5mm jack |

---

## References

- Hartley & Zisserman, *Multiple View Geometry in Computer Vision*, 2003
- Kirillov et al., *Segment Anything*, ICCV 2023
- Schönberger & Frahm, *Structure-from-Motion Revisited*, CVPR 2016
