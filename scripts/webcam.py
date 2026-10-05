"""Real-time anti-spoofing with OpenCV YuNet and temporal smoothing."""
from __future__ import annotations

import argparse
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from fas_core import load_checkpoint, transforms

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


def args():
    root = Path(__file__).parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=root / "artifacts" / "fas_mobilenetv3.pt")
    p.add_argument("--detector", type=Path, default=root / "models" / "face_detection_yunet_2023mar.onnx")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--threshold", type=float)
    p.add_argument("--margin", type=float, default=0.25)
    p.add_argument("--smoothing", type=int, default=8)
    return p.parse_args()


def ensure_detector(path: Path):
    if path.exists(): return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading YuNet to {path} ...")
    req = urllib.request.Request(YUNET_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(path, "wb") as f:
        f.write(resp.read())


def crop(frame, box, margin):
    x, y, w, h = box[:4].astype(int)
    pad = int(max(w, h) * margin)
    ih, iw = frame.shape[:2]
    return frame[max(0, y-pad):min(ih, y+h+pad), max(0, x-pad):min(iw, x+w+pad)]


def main():
    cfg = args(); ensure_detector(cfg.detector)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_checkpoint(cfg.checkpoint, device)
    real_idx = meta["class_to_idx"]["real"]
    threshold = cfg.threshold if cfg.threshold is not None else meta.get("real_threshold", 0.5)
    preprocess = transforms(False)
    detector = cv2.FaceDetectorYN.create(str(cfg.detector), "", (320, 320), 0.75, 0.3, 5000)
    cap = cv2.VideoCapture(cfg.camera)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open camera {cfg.camera}")
    history = deque(maxlen=cfg.smoothing)
    last = time.perf_counter(); fps = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            detector.setInputSize((frame.shape[1], frame.shape[0]))
            _, faces = detector.detect(frame)
            if faces is not None:
                # Single-user liveness: classify the largest face and reset smoothing if absent.
                face = max(faces, key=lambda f: f[2] * f[3])
                roi = crop(frame, face, cfg.margin)
                if roi.size:
                    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    x = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(device)
                    with torch.inference_mode():
                        real_prob = model(x).softmax(1)[0, real_idx].item()
                    history.append(real_prob)
                    score = float(np.mean(history))
                    label, color = ("REAL", (0, 200, 0)) if score >= threshold else ("SPOOF", (0, 0, 255))
                    bx, by, bw, bh = face[:4].astype(int)
                    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), color, 2)
                    cv2.putText(frame, f"{label} real={score:.2f}", (bx, max(25, by-8)), cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2)
            else:
                history.clear()
            now = time.perf_counter(); instant = 1 / max(now-last, 1e-6); last = now
            fps = instant if not fps else .9 * fps + .1 * instant
            cv2.putText(frame, f"FPS {fps:.1f} | q: quit", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
            cv2.imshow("Face anti-spoofing", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
    finally:
        cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

