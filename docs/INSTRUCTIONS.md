# Running Instructions

## Environment Setup

```bash
# 1. Create conda environment
conda create -n robotics_proj python=3.10
conda activate robotics_proj

# 2. Install PyTorch with CUDA 12.1 support (matches RTX 4090 / driver 535.x)
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining dependencies
pip install -r requirements.txt
```

## Data Setup

Place the dataset files under `robotics_proj_2/Data/`:
```
Data/
├── frame_000319.png
├── frame_000333.png
├── ...  (16 PNG frames in total)
├── frame_000531.png
├── poses.json       # camera-to-world 4×4 matrices, keyed by frame number
└── intrinsic.json   # camera intrinsics (fx, fy, cx, cy)
```

## Generate Predictions

```bash
cd robotics_proj_2

# Run pipeline for all registered entities
python src/pipeline.py

# Run for specific entities only
python src/pipeline.py --entities power_socket ethernet_socket
```

Output is written to `outputs/answers.json`.

## Visualise Results

```bash
python3 - << 'EOF'
import sys; sys.path.insert(0, 'src')
import config, utils, cv2, numpy as np, json
from pathlib import Path

poses = utils.load_poses()
K = utils.build_K()

with open('outputs/answers.json') as f:
    answers = json.load(f)

colors = [(0,200,0), (0,200,200), (0,100,255), (255,165,0), (255,0,255)]
val_dir = Path('outputs/validation'); val_dir.mkdir(exist_ok=True)

for frame_id in utils.get_frame_numbers():
    img = utils.load_image_bgr(frame_id)
    c2w = poses[frame_id]
    for i, entry in enumerate(answers):
        img = utils.draw_obb_on_image(img, entry['obb'], c2w, K,
                                       label=entry['entity'][:4],
                                       color=colors[i % len(colors)])
    cv2.imwrite(str(val_dir / f"frame_{frame_id:06d}.jpg"),
                cv2.resize(img, (1280, 720)))

print("Saved to outputs/validation/")
EOF
```

## Output Format

`outputs/answers.json` contains a list of OBB records:
```json
[
  {
    "entity": "ethernet_socket",
    "obb": {
      "center":   [x, y, z],
      "extent":   [half_e0, half_e1, half_e2],
      "rotation": [[...], [...], [...]]
    }
  }
]
```
