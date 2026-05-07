"""
Run simulation/main_abs.py for every method subfolder inside a parent directory,
then concatenate results horizontally.

Usage (from project root, with venv active):
    python simulation/run_batch.py [--concat-only] [--show-labels] [<parent_dir>]

Config files:
  config.yml        – shared settings; add  use_proxddp: true  to use optimized freepose
  config_proxddp.yml – ProxDDP weights, python path, compare_freepose option
"""

import subprocess
import sys
import re
import yaml
import numpy as np
import imageio
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ── config loading ─────────────────────────────────────────────────────────────

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


def _load_proxddp_config():
    path = PROJECT_ROOT / "config_proxddp.yml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


# ── video helpers ──────────────────────────────────────────────────────────────

def _resample(frames, target_n):
    n = len(frames)
    if n == target_n:
        return frames
    indices = [int(round(i * (n - 1) / (target_n - 1))) for i in range(target_n)]
    return [frames[i] for i in indices]


def _make_label_bar(width, label, bar_h=55, font_size=40):
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
    """Stack videos side by side (resampled to same frame count) and save."""
    print(f"\nConcatenating {len(video_paths)} videos -> {out_path}")
    for p in video_paths:
        print(f"  {'[OK]' if p.exists() else '[MISSING]'} {p.name}")

    existing_pairs = [(p, (labels[i] if labels else p.stem))
                      for i, p in enumerate(video_paths) if p.exists()]
    if not existing_pairs:
        print("  No videos found, skipping.")
        return
    existing = [p for p, _ in existing_pairs]
    names    = [n for _, n in existing_pairs]

    all_frames = [imageio.mimread(str(p), memtest=False) for p in existing]
    n_frames   = max(len(f) for f in all_frames)
    all_frames = [_resample(f, n_frames) for f in all_frames]

    h_ref  = all_frames[0][0].shape[0]
    resized = []
    for frames in all_frames:
        h, w = frames[0].shape[:2]
        if h != h_ref:
            import cv2
            new_w = int(w * h_ref / h)
            frames = [cv2.resize(f, (new_w, h_ref)) for f in frames]
        resized.append(frames)

    if show_labels:
        labeled = []
        for frames, name in zip(resized, names):
            w   = frames[0].shape[1]
            bar = _make_label_bar(w, name)
            labeled.append([np.concatenate([f, bar], axis=0) for f in frames])
        resized = labeled

    combined = [np.concatenate([v[i] for v in resized], axis=1) for i in range(n_frames)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out_path), combined, fps=fps, codec="libx264",
                     output_params=["-crf", "18"])
    print(f"  Saved -> {out_path}")


# ── ProxDDP optimization ───────────────────────────────────────────────────────

def _run_proxddp(proxddp_cfg: dict, data_dir: Path) -> Path:
    """Run optimize_traj.py on <data_dir>/object_pose/poses.npz.

    Writes poses_proxddp.npz to the same folder and returns its path.
    """
    proxddp_python  = proxddp_cfg.get("proxddp_python",
                                      "/home/dohyeon/miniconda3/envs/proxddp2/bin/python")
    optimize_script = Path(__file__).parent / "optimize_traj.py"
    poses_in  = data_dir / "object_pose" / "poses.npz"
    poses_out = data_dir / "object_pose" / "poses_proxddp.npz"

    cmd = [
        proxddp_python, str(optimize_script),
        str(poses_in), str(poses_out),
        "--w-d",   str(proxddp_cfg.get("w_d",   10.0)),
        "--w-dq",  str(proxddp_cfg.get("w_dq",  0.01)),
        "--w-tau", str(proxddp_cfg.get("w_tau",  0.001)),
        "--dt",    str(proxddp_cfg.get("dt",     0.1)),
    ]
    print(f"[proxddp] Optimizing {data_dir.name} ...")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"ProxDDP optimization failed for {data_dir}")
    return poses_out


# ── simulation runner ──────────────────────────────────────────────────────────

def _build_base_cmd(script, rel_data, video_base, gt_dir, is_gt):
    """Build the common main_abs.py command for a given subdir."""
    cmd = [
        sys.executable, str(script),
        str(rel_data),
        "--video-dir", str(video_base.relative_to(PROJECT_ROOT)),
    ]
    if gt_dir is not None:
        try:
            gt_rel = str(gt_dir.relative_to(PROJECT_ROOT))
        except ValueError:
            gt_rel = str(gt_dir)
        cmd += ["--gt-dir", gt_rel, "--mesh-dir", gt_rel]
    if not is_gt:
        cmd += ["--no-traj-video"]
    return cmd


def _run_sim(cmd, label, failed):
    print(f"{'='*60}")
    print(f"Running: {label}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    ok = result.returncode == 0
    print(f"[{'OK' if ok else 'FAILED'}] {label}")
    print()
    if not ok:
        failed.append(label)
    return ok


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    cfg          = _load_config()
    proxddp_cfg  = _load_proxddp_config()

    concat_only      = "--concat-only" in sys.argv or cfg.get("concat_only", False)
    show_labels      = "--show-labels" in sys.argv or cfg.get("show_labels", False)
    use_proxddp      = cfg.get("use_proxddp", False)
    compare_freepose = proxddp_cfg.get("compare_freepose", False)

    clean_argv = [a for a in sys.argv[1:] if a not in ("--concat-only", "--show-labels")]

    if not clean_argv:
        parent_str = cfg.get("batch_dir") or cfg.get("data_dir")
        if parent_str is None:
            print("Usage: python run_batch.py [--concat-only] [--show-labels] <parent_dir>")
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

    concat_order    = cfg.get("concat_order") or [d.name for d in subdirs]
    label_overrides = cfg.get("label_overrides") or {}

    def _label(name):
        return label_overrides.get(name, name)

    print(f"Parent dir       : {parent_dir}")
    print(f"Video base       : {video_base}")
    print(f"GT dir           : {gt_dir or '(none)'}")
    print(f"Methods          : {[d.name for d in subdirs]}")
    print(f"Concat order     : {concat_order}")
    print(f"use_proxddp      : {use_proxddp}")
    print(f"compare_freepose : {compare_freepose}")
    print()

    if concat_only:
        print("--concat-only: skipping simulation runs.")
    else:
        script = Path(__file__).parent / "main_abs.py"
        failed = []

        for subdir in subdirs:
            rel_data = subdir.relative_to(PROJECT_ROOT)
            is_gt    = (gt_dir is not None and subdir.resolve() == gt_dir.resolve())
            is_fp    = (subdir.name == "freepose")
            base_cmd = _build_base_cmd(script, rel_data, video_base, gt_dir, is_gt)

            needs_opt = is_fp and (use_proxddp or compare_freepose)

            if needs_opt:
                try:
                    _run_proxddp(proxddp_cfg, subdir)
                except Exception as e:
                    print(f"[FAILED] ProxDDP optimization: {e}")
                    failed.append(f"{subdir.name}(proxddp)")

            if is_fp and compare_freepose:
                # Run original  → freepose_orig.mp4
                _run_sim(base_cmd + ["--name", "freepose_orig"],
                         "freepose_orig", failed)
                # Run optimized → freepose_proxddp.mp4
                _run_sim(base_cmd + ["--poses-file", "poses_proxddp.npz",
                                     "--name", "freepose_proxddp"],
                         "freepose_proxddp", failed)
            elif is_fp and use_proxddp:
                # Normal batch run with optimized poses → freepose.mp4
                _run_sim(base_cmd + ["--poses-file", "poses_proxddp.npz"],
                         subdir.name, failed)
            else:
                _run_sim(base_cmd, subdir.name, failed)

        print("=" * 60)
        if failed:
            print(f"Runs done. Failed: {failed}")
        else:
            print(f"All runs completed.")

    # ── resolve which file represents "freepose" in the main concat ───────────
    def _sim_path(name):
        if name == "freepose" and compare_freepose:
            # use optimized side if use_proxddp, else original side
            stem = "freepose_proxddp" if use_proxddp else "freepose_orig"
            return video_base / f"{stem}.mp4"
        return video_base / f"{name}.mp4"

    # ── concat: traj + ALL sim videos ─────────────────────────────────────────
    gt_name         = gt_dir.name if gt_dir else None
    sim_paths       = [_sim_path(n) for n in concat_order]
    sim_labels      = [_label(n) for n in concat_order]
    traj_first      = video_base / f"{gt_name}_traj.mp4" if gt_name else None
    traj_all_paths  = ([traj_first] if traj_first else []) + sim_paths
    traj_all_labels = (["rgb"] if traj_first else []) + sim_labels
    concat_horizontal(traj_all_paths, video_base / "concat_traj_all.mp4",
                      show_labels=show_labels, labels=traj_all_labels)

    # ── compare_freepose: original | optimized side-by-side ───────────────────
    if compare_freepose:
        fp_orig = video_base / "freepose_orig.mp4"
        fp_opt  = video_base / "freepose_proxddp.mp4"
        concat_horizontal([fp_orig, fp_opt], video_base / "freepose_compare.mp4",
                          show_labels=show_labels,
                          labels=["freepose (original)", "freepose (proxddp)"])

    print("\nAll done.")


if __name__ == "__main__":
    main()
