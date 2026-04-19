# Usage:
#   python document/mp4_to_gif.py dataset_cam.mp4                           
#   python document/mp4_to_gif.py 006_mustard_bottle_20200709_143211/dataset_cam.mp4  
#   python document/mp4_to_gif.py videos/006_mustard_bottle_20200709_143211/dataset_cam.mp4 
#   python document/mp4_to_gif.py clip.mp4 --width 360 --every 3 --colors 128
#   python document/mp4_to_gif.py clip.mp4 --out custom_name.gif

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import imageio
import numpy as np
import PIL.Image


def mp4_to_gif(src: Path, out: Path, width: int, every: int, colors: int):
    reader = imageio.get_reader(src)
    fps = reader.get_meta_data().get("fps", 20)

    frames = []
    for i, frame in enumerate(reader):
        if i % every != 0:
            continue
        h, w = frame.shape[:2]
        new_h = int(h * width / w)
        img = PIL.Image.fromarray(frame).resize((width, new_h), PIL.Image.LANCZOS)
        img = img.quantize(colors=colors, method=PIL.Image.Quantize.MEDIANCUT)
        frames.append(img)
    reader.close()

    duration_ms = int(1000 / (fps / every))
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration_ms,
        optimize=True,
    )
    print(f"Saved {out}  ({out.stat().st_size / 1024 / 1024:.1f} MB,  {len(frames)} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src",            help="input .mp4 file")
    parser.add_argument("--out",          default=None, help="output .gif path (default: same name)")
    parser.add_argument("--width",  type=int, default=360,  help="output width in pixels")
    parser.add_argument("--every",  type=int, default=3,    help="keep every Nth frame")
    parser.add_argument("--colors", type=int, default=128,  help="palette size (max 256)")
    args = parser.parse_args()

    src = Path(args.src)
    if not src.is_absolute():
        candidates = [
            PROJECT_ROOT / src,                   # exact relative path from project root
            PROJECT_ROOT / "videos" / src,         # relative path under videos/
        ]
        found = next((c for c in candidates if c.exists()), None)
        if found is None:
            # last resort: search by filename under videos/**/
            matches = list((PROJECT_ROOT / "videos").rglob(src.name))
            if not matches:
                raise FileNotFoundError(f"Could not find '{src}' under {PROJECT_ROOT}")
            if len(matches) > 1:
                print("Multiple matches, using first:")
                for m in matches:
                    print(f"  {m}")
            found = matches[0]
        src = found
    out = Path(args.out) if args.out else PROJECT_ROOT / "document" / src.with_suffix(".gif").name
    mp4_to_gif(src, out, args.width, args.every, args.colors)
