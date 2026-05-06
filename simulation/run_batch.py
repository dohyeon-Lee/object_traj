"""
Run simulation/main_abs.py for every method subfolder inside a parent directory.

Usage (from project root, with venv active):
    python simulation/run_batch.py data/dexycb/20200820_144100

Config (config.yml) is shared as-is. Only data_dir, video_dir, and gt_dir are
overridden per run. If gt_dir is set in config, that specific subfolder is used
as the camera reference for ALL runs.
"""

import subprocess
import sys
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_config():
    path = PROJECT_ROOT / "config.yml"
    if path.exists():
        return yaml.safe_load(path.read_text()) or {}
    return {}


def main():
    cfg = _load_config()

    if len(sys.argv) < 2:
        parent_str = cfg.get("batch_dir") or cfg.get("data_dir")
        if parent_str is None:
            print("Usage: python run_batch.py <parent_dir>")
            sys.exit(1)
        # strip last component if it's already a method folder
        parent_dir = PROJECT_ROOT / parent_str
        if not any(parent_dir.iterdir().__next__().is_dir() for _ in [None]):
            parent_dir = parent_dir.parent
    else:
        parent_dir = Path(sys.argv[1])
        if not parent_dir.is_absolute():
            parent_dir = PROJECT_ROOT / parent_dir

    if not parent_dir.exists():
        print(f"Error: {parent_dir} does not exist")
        sys.exit(1)

    subdirs = sorted([d for d in parent_dir.iterdir() if d.is_dir()])
    if not subdirs:
        print(f"No subdirectories found in {parent_dir}")
        sys.exit(1)

    # Determine gt_dir: use config value if set, else None
    gt_dir_cfg = cfg.get("gt_dir")
    if gt_dir_cfg:
        gt_dir = Path(gt_dir_cfg)
        if not gt_dir.is_absolute():
            gt_dir = PROJECT_ROOT / gt_dir
    else:
        gt_dir = None

    # video_dir base: strip the method-level folder from config video_dir if present,
    # or just use parent_dir's relative path structure under "videos/"
    cfg_video_dir = cfg.get("video_dir")
    if cfg_video_dir:
        video_base = Path(cfg_video_dir)
        if not video_base.is_absolute():
            video_base = PROJECT_ROOT / video_base
        # Go up one level if it already ends with a method name that matches a subdir
        if video_base.name in [d.name for d in subdirs]:
            video_base = video_base.parent
    else:
        # Default: videos/<relative parent path>
        try:
            rel = parent_dir.relative_to(PROJECT_ROOT)
        except ValueError:
            rel = Path(parent_dir.name)
        video_base = PROJECT_ROOT / "videos" / rel

    print(f"Parent dir : {parent_dir}")
    print(f"Video base : {video_base}")
    print(f"GT dir     : {gt_dir or '(none)'}")
    print(f"Methods    : {[d.name for d in subdirs]}")
    print()

    script = Path(__file__).parent / "main_abs.py"
    failed = []

    for subdir in subdirs:
        rel_data = subdir.relative_to(PROJECT_ROOT)
        video_dir = video_base  # flat: all go to same base, filenames differ by run_name

        cmd = [
            sys.executable, str(script),
            str(rel_data),
            "--video-dir", str(video_dir.relative_to(PROJECT_ROOT)),
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
        print(f"Done. Failed: {failed}")
    else:
        print(f"Done. All {len(subdirs)} runs completed successfully.")


if __name__ == "__main__":
    main()
