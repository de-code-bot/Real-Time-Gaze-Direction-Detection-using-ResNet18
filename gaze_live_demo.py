"""
gaze_live_demo.py
=================
Standalone live gaze detection — no retraining needed.
Loads your saved model and runs real-time inference from webcam.

Gaze is classified into 8 directions:
    UP-LEFT   |   UP   |   UP-RIGHT
    LEFT      | CENTER |   RIGHT
    DOWN-LEFT |  DOWN  |  DOWN-RIGHT

Usage:
    python gaze_live_demo.py
    python gaze_live_demo.py --model gaze_model.pth
    python gaze_live_demo.py --model gaze_model.pth --camera 0 --img_size 60

Press Q or ESC to quit.
Press S to save a screenshot.
Press R to reset the direction history/counts.
"""

import argparse
import time
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Live Gaze Detection Demo')
parser.add_argument('--model',    type=str, default='gaze_model.pth',
                    help='Path to saved model weights (.pth)')
parser.add_argument('--camera',   type=int, default=0,
                    help='Webcam index (default: 0)')
parser.add_argument('--img_size', type=int, default=60,
                    help='Image size used during training (default: 60)')
parser.add_argument('--threshold_pitch', type=float, default=0.08,
                    help='Pitch threshold in radians to classify UP/DOWN (default: 0.08 ~4.6 deg)')
parser.add_argument('--threshold_yaw',   type=float, default=0.10,
                    help='Yaw threshold in radians to classify LEFT/RIGHT (default: 0.10 ~5.7 deg)')
args = parser.parse_args()


# ── Model definition (must match training exactly) ───────────────────────────
class GazeEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.features  = nn.Sequential(*list(backbone.children())[:-1])
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.regressor(self.features(x))


# ── Load model ────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(args.model):
    print(f'ERROR: Model file not found: {args.model}')
    print('Make sure gaze_model.pth is in the same folder as this script,')
    print('or pass the correct path with --model /path/to/gaze_model.pth')
    sys.exit(1)

print(f'Loading model from: {args.model}')
model = GazeEstimator().to(DEVICE)

ckpt = torch.load(args.model, map_location=DEVICE)
# Handle both bare state_dict and full checkpoint dict
if isinstance(ckpt, dict) and 'model_state' in ckpt:
    model.load_state_dict(ckpt['model_state'])
else:
    model.load_state_dict(ckpt)

model.eval()
print(f'Model loaded successfully on {DEVICE}')


# ── Image preprocessing ───────────────────────────────────────────────────────
TRANSFORM = T.Compose([
    T.Resize((args.img_size, args.img_size)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


def preprocess_eye(eye_gray):
    """Convert grayscale eye ROI numpy array → model input tensor."""
    pil = Image.fromarray(eye_gray).convert('RGB')
    return TRANSFORM(pil).unsqueeze(0).to(DEVICE)


# ── Gaze direction classification ─────────────────────────────────────────────
PITCH_TH = args.threshold_pitch   # radians — positive = looking down
YAW_TH   = args.threshold_yaw     # radians — positive = looking right

# Direction label, display color (BGR), compass symbol
DIRECTIONS = {
    'CENTER'     : ('CENTER',     (200, 200, 200), ' + '),
    'UP'         : ('UP',         ( 80, 220, 255), ' ^ '),
    'DOWN'       : ('DOWN',       ( 80, 220, 255), ' v '),
    'LEFT'       : ('LEFT',       (255, 180,  50), ' < '),
    'RIGHT'      : ('RIGHT',      (255, 180,  50), ' > '),
    'UP-LEFT'    : ('UP-LEFT',    (120, 255, 150), '/^ '),
    'UP-RIGHT'   : ('UP-RIGHT',   (120, 255, 150), ' ^\\'),
    'DOWN-LEFT'  : ('DOWN-LEFT',  (100, 180, 255), '\\v '),
    'DOWN-RIGHT' : ('DOWN-RIGHT', (100, 180, 255), ' v/'),
}


def classify_gaze(pitch, yaw):
    """
    Map (pitch, yaw) in radians to a gaze direction label.

    Convention (MPIIGaze):
        pitch > 0  → looking DOWN
        pitch < 0  → looking UP
        yaw   > 0  → looking RIGHT
        yaw   < 0  → looking LEFT
    """
    up    = pitch < -PITCH_TH
    down  = pitch >  PITCH_TH
    left  = yaw   < -YAW_TH
    right = yaw   >  YAW_TH

    if   up    and left:  return 'UP-LEFT'
    elif up    and right: return 'UP-RIGHT'
    elif down  and left:  return 'DOWN-LEFT'
    elif down  and right: return 'DOWN-RIGHT'
    elif up:              return 'UP'
    elif down:            return 'DOWN'
    elif left:            return 'LEFT'
    elif right:           return 'RIGHT'
    else:                 return 'CENTER'


# ── Gaze arrow drawing ────────────────────────────────────────────────────────
def draw_gaze_arrow(frame, center, pitch, yaw, length=90, color=(50, 255, 100)):
    dx = int(-length * np.cos(pitch) * np.sin(yaw))
    dy = int(-length * np.sin(pitch))
    x0, y0 = int(center[0]), int(center[1])
    cv2.arrowedLine(frame, (x0, y0), (x0 + dx, y0 + dy),
                    color, 2, tipLength=0.35)
    cv2.circle(frame, (x0, y0), 4, color, -1)


# ── Direction compass overlay ─────────────────────────────────────────────────
COMPASS_POSITIONS = {
    'UP-LEFT'   : (0, 0), 'UP'    : (1, 0), 'UP-RIGHT'   : (2, 0),
    'LEFT'      : (0, 1), 'CENTER': (1, 1), 'RIGHT'       : (2, 1),
    'DOWN-LEFT' : (0, 2), 'DOWN'  : (1, 2), 'DOWN-RIGHT'  : (2, 2),
}

def draw_compass(frame, active_direction, origin=(20, 100)):
    """Draw a 3x3 compass grid showing active gaze direction."""
    ox, oy   = origin
    cell_sz  = 42
    pad      = 4

    for label, (col, row) in COMPASS_POSITIONS.items():
        x = ox + col * (cell_sz + pad)
        y = oy + row * (cell_sz + pad)

        is_active = (label == active_direction)
        bg_color  = DIRECTIONS[label][1] if is_active else (40, 40, 40)
        bd_color  = DIRECTIONS[label][1] if is_active else (80, 80, 80)

        cv2.rectangle(frame, (x, y), (x + cell_sz, y + cell_sz), bg_color, -1)
        cv2.rectangle(frame, (x, y), (x + cell_sz, y + cell_sz), bd_color,  1)

        short = label.replace('-', '\n')
        lines = short.split('\n')
        font  = cv2.FONT_HERSHEY_SIMPLEX
        for i, line in enumerate(lines):
            fs   = 0.32
            tw, th = cv2.getTextSize(line, font, fs, 1)[0]
            tx   = x + (cell_sz - tw) // 2
            ty   = y + (cell_sz - len(lines) * (th + 2)) // 2 + (i + 1) * (th + 2)
            col  = (10, 10, 10) if is_active else (160, 160, 160)
            cv2.putText(frame, line, (tx, ty), font, fs, col, 1, cv2.LINE_AA)


# ── Direction history (smoothing) ─────────────────────────────────────────────
from collections import deque, Counter

SMOOTH_WINDOW  = 8   # number of recent frames to smooth over
direction_buf  = deque(maxlen=SMOOTH_WINDOW)
direction_counts = {k: 0 for k in DIRECTIONS}


def smoothed_direction(new_dir):
    direction_buf.append(new_dir)
    return Counter(direction_buf).most_common(1)[0][0]


# ── Haar cascades ─────────────────────────────────────────────────────────────
HAAR = cv2.data.haarcascades
face_cascade = cv2.CascadeClassifier(os.path.join(HAAR, 'haarcascade_frontalface_default.xml'))
eye_cascade  = cv2.CascadeClassifier(os.path.join(HAAR, 'haarcascade_eye.xml'))

if face_cascade.empty() or eye_cascade.empty():
    print('ERROR: OpenCV Haar cascades not found. Reinstall opencv-python.')
    sys.exit(1)


# ── Open camera ───────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    print(f'ERROR: Cannot open camera index {args.camera}.')
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
cap.set(cv2.CAP_PROP_FPS, 30)

FONT         = cv2.FONT_HERSHEY_DUPLEX
screenshot_n = 0
prev_time    = time.time()

print('\nCamera opened. Starting live demo.')
print('  Q / ESC  — quit')
print('  S        — screenshot')
print('  R        — reset direction counts')
print()

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print('Failed to read frame. Exiting.')
        break

    # FPS
    now      = time.time()
    fps      = 1.0 / max(now - prev_time, 1e-6)
    prev_time = now

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )

    frame_directions = []

    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 200, 50), 2)
        cv2.putText(frame, 'face', (fx, fy - 6),
                    FONT, 0.4, (255, 200, 50), 1, cv2.LINE_AA)

        roi_gray  = gray [fy:fy+fh, fx:fx+fw]
        roi_color = frame[fy:fy+fh, fx:fx+fw]

        eyes = eye_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.05, minNeighbors=4, minSize=(35, 35)
        )

        for (ex, ey, ew, eh) in eyes:
            eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
            if eye_roi.size == 0:
                continue

            with torch.no_grad():
                inp  = preprocess_eye(eye_roi)
                pred = model(inp).cpu().numpy()[0]

            pitch, yaw = float(pred[0]), float(pred[1])
            direction  = classify_gaze(pitch, yaw)
            frame_directions.append(direction)

            arrow_color = DIRECTIONS[direction][1]
            cx = fx + ex + ew // 2
            cy = fy + ey + eh // 2

            draw_gaze_arrow(frame, (cx, cy), pitch, yaw,
                            color=arrow_color)

            cv2.rectangle(roi_color,
                          (ex, ey), (ex+ew, ey+eh),
                          (50, 220, 255), 1)
            cv2.putText(frame,
                        f'P:{np.degrees(pitch):+.1f} Y:{np.degrees(yaw):+.1f}',
                        (fx + ex, fy + ey - 5),
                        FONT, 0.36, (50, 220, 255), 1, cv2.LINE_AA)

    # ── Determine dominant direction this frame ───────────────────────────────
    if frame_directions:
        # Average across both eyes if multiple detected
        dominant = Counter(frame_directions).most_common(1)[0][0]
    else:
        dominant = 'CENTER'

    smooth_dir = smoothed_direction(dominant)
    direction_counts[smooth_dir] += 1

    # ── Draw compass ──────────────────────────────────────────────────────────
    draw_compass(frame, smooth_dir, origin=(20, 90))

    # ── Big direction label ───────────────────────────────────────────────────
    dir_color = DIRECTIONS[smooth_dir][1]
    symbol    = DIRECTIONS[smooth_dir][2]

    # Background pill for direction label
    label_text = f'{symbol}  {smooth_dir}  {symbol}'
    (tw, th), _ = cv2.getTextSize(label_text, FONT, 1.1, 2)
    lx = frame.shape[1] // 2 - tw // 2
    ly = frame.shape[0] - 60
    cv2.rectangle(frame, (lx - 12, ly - th - 8),
                  (lx + tw + 12, ly + 8), (20, 20, 20), -1)
    cv2.rectangle(frame, (lx - 12, ly - th - 8),
                  (lx + tw + 12, ly + 8), dir_color, 2)
    cv2.putText(frame, label_text, (lx, ly),
                FONT, 1.1, dir_color, 2, cv2.LINE_AA)

    # ── HUD top-right ─────────────────────────────────────────────────────────
    hud = [
        f'FPS   : {fps:.1f}',
        f'Faces : {len(faces)}',
        f'Device: {str(DEVICE).upper()}',
    ]
    for i, line in enumerate(hud):
        cv2.putText(frame, line,
                    (frame.shape[1] - 220, 28 + i * 24),
                    FONT, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    # ── Direction count bar (bottom right) ────────────────────────────────────
    total_counts = sum(direction_counts.values()) or 1
    bar_x  = frame.shape[1] - 220
    bar_y  = 110
    cv2.putText(frame, 'Direction counts:', (bar_x, bar_y),
                FONT, 0.42, (130, 130, 130), 1, cv2.LINE_AA)
    for i, (dname, cnt) in enumerate(direction_counts.items()):
        pct   = cnt / total_counts * 100
        color = DIRECTIONS[dname][1] if cnt > 0 else (60, 60, 60)
        text  = f'{dname:<12} {pct:4.1f}%'
        cv2.putText(frame, text,
                    (bar_x, bar_y + 20 + i * 18),
                    FONT, 0.38, color, 1, cv2.LINE_AA)

    # ── Key hint ──────────────────────────────────────────────────────────────
    cv2.putText(frame, 'Q:quit  S:screenshot  R:reset',
                (12, frame.shape[0] - 12),
                FONT, 0.45, (100, 100, 100), 1, cv2.LINE_AA)

    cv2.imshow('Gaze Direction Detection', frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        break
    elif key == ord('s'):
        fname = f'gaze_screenshot_{screenshot_n:03d}.png'
        cv2.imwrite(fname, frame)
        print(f'Screenshot saved: {fname}')
        screenshot_n += 1
    elif key == ord('r'):
        direction_counts = {k: 0 for k in DIRECTIONS}
        direction_buf.clear()
        print('Direction counts reset.')


cap.release()
cv2.destroyAllWindows()
print('Done.')
