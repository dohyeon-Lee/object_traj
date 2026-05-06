"""
Run simulation/main_abs.py for every method subfolder inside a parent directory,
then concatenate results horizontally.

Usage (from project root, with venv active):
    python simulation/run_batch.py data/dexycb/20200820_144100

Config (config.yml) is shared as-is. Only data_dir, video_dir, and gt_dir are
overridden per run. If gt_dir is set in config, that specific subfolder is used
as the camera reference for ALL runs.
"""

import subprocess
import sys
import re
import yaml
import numpy as np
import imageio
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_config():
    path = PROJECT_ROOT / "config.yml"
    if not path.exists():
        return {}
    cfg = yaml.safe_load(path.read_text()) or {}
    def _sub(v):
        if not isinstance(v, str):
            return v
        return re.sub(r'\$\{(\w+)\}', lambda m: str(cfg.get(m.group(1), m.group(0))), v)
    return {k: _sub(v) for k, v in cfg.items()}


def _resample(frames, target_n):
    """Nearest-neighbor temporal resample to target_n frames."""
    n = len(frames)
    if n == target_n:
        return frames
    indices = [int(round(i * (n - 1) / (target_n - 1))) for i in range(target_n)]
    return [frames[i] for i in indices]


def _make_label_bar(width, label, bar_h=55, font_size=40):
    """Black bar with centered white label text."""
    from PIL import Image, ImageDraw, ImageFont
    bar = Image.new("RGB", (width, bar_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(bar)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - tw) // 2, (bar_h - th) // 2), label, fill=(255, 255, 255), font=font)
    return np.array(bar)


def concat_horizontal(video_paths, out_path, fps=20, show_labels=False, labels=None):
    """Stack videos side by side, resampled to the same frame count, and save."""
    print(f"\nConcatenating {len(video_paths)} videos -> {out_path}")
    for p in video_paths:
        print(f"  {'[OK]' if p.exists() else '[MISSING]'} {p.name}")

    existing_pairs = [(p, (labels[i] if labels else p.stem)) for i, p in enumerate(video_paths) if p.exists()]
    if not existing_pairs:
        print("  No videos found, skipping.")
        return
    existing = [p for p, _ in existing_pairs]
    names    = [n for _, n in existing_pairs]

    all_frames = [imageio.mimread(str(p), memtest=False) for p in existing]

    # Use maximum frame count as target so no video is clipped
    n_frames = max(len(f) for f in all_frames)
    all_frames = [_resample(f, n_frames) for f in all_frames]

    # Resize all to same height (use first video's height as reference)
    h_ref = all_frames[0][0].shape[0]
    resized = []
    for frames in all_frames:
        h, w = frames[0].shape[:2]
        if h != h_ref:
            import cv2
            scale = h_ref / h
            new_w = int(w * scale)
            frames = [cv2.resize(f, (new_w, h_ref)) for f in frames]
        resized.append(frames)

    # Optionally attach label bar below each video column
    if show_labels:
        labeled = []
        for frames, name in zip(resized, names):
            w = frames[0].shape[1]
            bar = _make_label_bar(w, name)
            labeled.append([np.concatenate([f, bar], axis=0) for f in frames])
        resized = labeled

    combined = [np.concatenate([v[i] for v in resized], axis=1) for i in range(n_frames)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out_path), combined, fps=fps, codec="libx264",
                     output_params=["-crf", "18"])
    print(f"  Saved -> {out_path}")


def main():
    cfg = _load_config()
    concat_only  = "--concat-only"  in sys.argv or cfg.get("concat_only", False)
    show_labels  = "--show-labels"  in sys.argv or cfg.get("show_labels", False)
    clean_argv = [a for a in sys.argv[1:] if a not in ("--concat-only", "--show-labels")]

    if not clean_argv:
        parent_str = cfg.get("batch_dir") or cfg.get("data_dir")
        if parent_str is None:
            print("Usage: python run_batch.py [--concat-only] <parent_dir>")
            sys.exit(1)
        parent_dir = PROJECT_ROOT / parent_str
        if not parent_dir.is_dir() or not any(d.is_dir() for d in parent_dir.iterdir()):
            parent_dir = parent_dir.parent
    else:
        parent_dir = Path(clean_argv[0])
        if not parent_dir.is_absolute():
            parent_dir = PROJECT_ROOT / parent_dir

    if not parent_dir.exists():
        print(f"Error: {parent_dir} does not exist")
        sys.exit(1)

    subdirs = sorted([d for d in parent_dir.iterdir() if d.is_dir()])
    if not subdirs:
        print(f"No subdirectories found in {parent_dir}")
        sys.exit(1)

    gt_dir_cfg = cfg.get("gt_dir")
    if gt_dir_cfg:
        gt_dir = Path(gt_dir_cfg)
        if not gt_dir.is_absolute():
            gt_dir = PROJECT_ROOT / gt_dir
    else:
        gt_dir = None

    cfg_video_dir = cfg.get("video_dir")
    if cfg_video_dir:
        video_base = Path(cfg_video_dir)
        if not video_base.is_absolute():
            video_base = PROJECT_ROOT / video_base
        if video_base.name in [d.name for d in subdirs]:
            video_base = video_base.parent
    else:
        try:
            rel = parent_dir.relative_to(PROJECT_ROOT)
        except ValueError:
            rel = Path(parent_dir.name)
        video_base = PROJECT_ROOT / "videos" / rel

    # concat_order: from config or default alphabetical
    concat_order    = cfg.get("concat_order") or [d.name for d in subdirs]
    label_overrides = cfg.get("label_overrides") or {}

    def _label(name):
        return label_overrides.get(name, name)

    print(f"Parent dir   : {parent_dir}")
    print(f"Video base   : {video_base}")
    print(f"GT dir       : {gt_dir or '(none)'}")
    print(f"Methods      : {[d.name for d in subdirs]}")
    print(f"Concat order : {concat_order}")
    print()

    if concat_only:
        print("--concat-only: skipping simulation runs.")
    else:
        script = Path(__file__).parent / "main_abs.py"
        failed = []

        for subdir in subdirs:
            rel_data = subdir.relative_to(PROJECT_ROOT)

            cmd = [
                sys.executable, str(script),
                str(rel_data),
                "--video-dir", str(video_base.relative_to(PROJECT_ROOT)),
            ]
            if gt_dir is not None:
                try:
                    cmd += ["--gt-dir", str(gt_dir.relative_to(PROJECT_ROOT))]
                except ValueError:
                    cmd += ["--gt-dir", str(gt_dir)]

            is_gt = (gt_dir is not None and subdir.resolve() == gt_dir.resolve())
            if not is_gt:
                cmd += ["--no-traj-video"]

            print(f"{'='*60}")
            print(f"Running: {subdir.name}")
            print(f"  cmd: {' '.join(cmd)}")
            print(f"{'='*60}")

            result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            if result.returncode != 0:
                print(f"[FAILED] {subdir.name} (exit {result.returncode})")
                failed.append(subdir.name)
            else:
                print(f"[OK] {subdir.name}")
            print()

        print("=" * 60)
        if failed:
            print(f"Runs done. Failed: {failed}")
        else:
            print(f"All {len(subdirs)} runs completed.")

    # ── concat sim videos ──────────────────────────────────────────────────────
    gt_name = gt_dir.name if gt_dir else None
    sim_paths  = [video_base / f"{name}.mp4" for name in concat_order]
    sim_labels = [_label(n) for n in concat_order]
    concat_horizontal(sim_paths, video_base / "concat_sim.mp4",
                      show_labels=show_labels, labels=sim_labels)

    # ── concat traj + sim videos of non-gt methods (gt_traj | megapose | ... | ours) ──
    traj_first  = video_base / f"{gt_name}_traj.mp4" if gt_name else None
    rest_paths  = [video_base / f"{name}.mp4" for name in concat_order if name != gt_name]
    rest_labels = [_label(n) for n in concat_order if n != gt_name]
    traj_paths  = ([traj_first] if traj_first else []) + rest_paths
    traj_labels = (["rgb"] if traj_first else []) + rest_labels
    concat_horizontal(traj_paths, video_base / "concat_traj.mp4",
                      show_labels=show_labels, labels=traj_labels)

    # ── concat traj + ALL sim videos (gt_traj | gt | megapose | ... | ours) ───
    traj_all_paths  = ([traj_first] if traj_first else []) + sim_paths
    traj_all_labels = (["rgb"] if traj_first else []) + sim_labels
    concat_horizontal(traj_all_paths, video_base / "concat_traj_all.mp4",
                      show_labels=show_labels, labels=traj_all_labels)

    print("\nAll done.")


if __name__ == "__main__":
    main()
