"""Local Real-ESRGAN 2x video upscaler. Mirrors cv_local/RealESR/upscale_video.py
but uses the on-disk weight (cv_local/RealESR/weights/realesr-general-x4v3.pth)
instead of the GitHub release URL — no network needed at runtime.
"""
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

from realesrgan import RealESRGANer
from basicsr.archs.srvgg_arch import SRVGGNetCompact

WEIGHT = Path("/home/chatsign/lizh/cv_local/RealESR/weights/realesr-general-x4v3.pth")

inp = sys.argv[1]
out = sys.argv[2]
outscale = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0

if not WEIGHT.exists():
    sys.exit(f"weight missing: {WEIGHT}")

model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_conv=32, upscale=4, act_type="prelu")
upsampler = RealESRGANer(
    scale=4, model_path=str(WEIGHT),
    model=model, half=True, device="cuda",
)

cap = cv2.VideoCapture(inp)
fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
oW, oH = int(W * outscale), int(H * outscale)

writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (oW, oH))
for _ in tqdm(range(n)):
    ok, frame = cap.read()
    if not ok:
        break
    sr, _ = upsampler.enhance(frame, outscale=outscale)
    writer.write(sr)
cap.release()
writer.release()
print(f"wrote {out}: {oW}x{oH} @ {fps}fps, {n} frames")
