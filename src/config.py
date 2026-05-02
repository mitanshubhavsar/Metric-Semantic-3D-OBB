"""
Configuration for the Metric-Semantic 3D Reconstruction Pipeline.
Scene: Desktop PC tower on a white pedestal, orbited by a camera.
Task: Identify 3D OBBs of back-panel ports (power socket, ethernet socket, etc.)
"""

import json
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR   = os.path.dirname(SRC_DIR)
DATA_DIR   = os.path.join(PROJ_DIR, "Data")
OUTPUT_DIR = os.path.join(PROJ_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Camera intrinsics (loaded from intrinsic.json) ─────────────────────────────
def _load_intrinsics():
    path = os.path.join(PROJ_DIR, "intrinsic.json")
    with open(path) as f:
        d = json.load(f)
    K = d["camera_matrix"]
    return (float(K[0][0]), float(K[1][1]),
            float(K[0][2]), float(K[1][2]),
            int(d["image_width"]), int(d["image_height"]))

FX, FY, CX, CY, IMAGE_W, IMAGE_H = _load_intrinsics()
# FX≈1477.0  FY≈1480.4  CX≈1298.25  CY≈686.82  W=2560  H=1440

# ── Ground-truth anchors (from sample_answers.json) ────────────────────────────
# VGA socket is provided as the reference example with exact OBB.
# All back-panel ports share the same face orientation → same rotation matrix.
VGA_CENTER   = [0.2704921202927293, 0.2261220732082181, 0.8349008829378597]
VGA_EXTENT   = [0.03537766175069747, 0.011822199241650923, 0.0061316691090621735]
VGA_ROTATION = [
    [-0.004004375172752437,  0.9672545151126772, -0.25377680739897346],
    [ 0.01584254528462312,   0.25380835519540434, 0.9671247761234889],
    [ 0.9998664804554559,   -0.00014774012094266402, -0.016340117333610394],
]
# Column 0 of VGA_ROTATION = panel normal (pointing outward toward camera)
# Column 1 = horizontal direction along the panel
# Column 2 = vertical direction along the panel

# ── Per-entity OWL-ViT text prompts ───────────────────────────────────────────
ENTITY_PROMPTS = {
    "vga_socket": [
        "vga port", "vga connector", "blue d-sub port",
        "blue trapezoidal port", "monitor port",
    ],
    "ethernet_socket": [
        "ethernet port", "rj45 port", "network socket",
        "lan port", "ethernet socket", "rj45 jack",
    ],
    "power_socket": [
        "power supply inlet", "iec power connector",
        "c14 power socket", "power inlet", "power socket",
        "iec c14 connector", "3-pin power inlet",
    ],
}

# ── Per-entity approximate 3D centers (priors for SAM prompt points) ──────────
# These come from visual triangulation in robotics_project/triangulate_visual.py.
# Used as initial anchor to project a prompt point into each frame for SAM.
ENTITY_PRIOR_CENTER = {
    "vga_socket":      VGA_CENTER,
    "ethernet_socket": [0.29270, 0.21600, 0.75620],
    "power_socket":    [0.29110, 0.21500, 0.52610],
    # Bonus ports (single-frame backprojection from frame 468, depth ≈ 0.622 m)
    "usb_socket":      [0.2700, 0.1999, 0.7738],   # USB 3.0 block (two stacked Type-A)
    "audio_socket":    [0.2669, 0.1932, 0.7335],   # 3.5mm line-out / speaker jack
}

# ── Per-entity physical full dimensions (width × height in metres) ─────────────
# These are the REAL physical sizes of the connectors on the back panel.
# IEC C14: 47mm × 30mm (outer housing)
# RJ-45:   16mm × 13mm (port opening)
# VGA D-Sub 15: GT-confirmed ~71mm × 24mm bounding region
ENTITY_PHYSICAL_WH_M = {
    "vga_socket":      (0.0708, 0.0236),   # from GT extent * 2
    "ethernet_socket": (0.018, 0.015),      # RJ45 full size
    "power_socket":    (0.050, 0.032),      # IEC C14 outer
}

# ── Calibrated OBB half-extents (metres) — used as fallback in pipeline ───────
# Calibrated from VGA GT: GT ≈ 0.77× physical horizontal, 0.97× physical vertical.
# Depth axis (axis0 = panel normal) = VGA GT value for all ports (triangulation smear).
# Format: [depth_half, horizontal_half, vertical_half]
VGA_DEPTH_HALF = VGA_EXTENT[0]   # 0.035378 m
ENTITY_CALIBRATED_EXTENT = {
    "vga_socket":      VGA_EXTENT,                                        # GT
    "ethernet_socket": [VGA_DEPTH_HALF, 0.006051, 0.006588],             # RJ45 body
    "power_socket":    [VGA_DEPTH_HALF, 0.011107, 0.009760],             # IEC C14 face
    "usb_socket":      [VGA_DEPTH_HALF, 0.005592, 0.011712],             # 2× USB-A stack
    "audio_socket":    [VGA_DEPTH_HALF, 0.003447, 0.004392],             # 3.5mm jack
}

# ── Detection / filtering parameters ─────────────────────────────────────────
OWL_SCORE_THRESHOLD   = 0.10
NMS_IOU_THRESHOLD     = 0.30
ANCHOR_RADIUS_PX      = 350    # max px distance from projected prior to accept detection
BOX_SIZE_MIN_RATIO    = 0.15   # min ratio of detected vs expected pixel size
BOX_SIZE_MAX_RATIO    = 5.0    # max ratio

# ── Reconstruction parameters ─────────────────────────────────────────────────
MIN_POINTS_FOR_OBB    = 20
OUTLIER_NB_NEIGHBORS  = 15
OUTLIER_STD_RATIO     = 2.5
DBSCAN_EPS_M          = 0.04   # 4cm neighbourhood radius for clustering
DBSCAN_MIN_SAMPLES    = 5

# ── SAM parameters ───────────────────────────────────────────────────────────
SAM_MODEL = "facebook/sam-vit-base"
# Radius (pixels) around the prompt point to consider for mask selection
SAM_PROMPT_RADIUS_PX  = 50
