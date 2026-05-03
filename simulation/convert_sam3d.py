#!/usr/bin/env python3
"""Convert SAM3D-format data dir to the standard format expected by main_abs.py / viz_overlay.py.

Requires:  pip install trimesh[easy]

Creates (in-place inside data_dir):
  object_pose/poses.npz   – (N,4,4) pose matrices from R_sm + t_sm (or --pose-R/--pose-t)
  camera.json             – height/width/intrinsics from meta.json
  mesh/textured_simple.obj – geometry from sam3d_mesh.glb, coord-transformed to CV frame
  mesh/texture_map.png    – placeholder gray texture (MuJoCo requires one)
  rgb/{:06d}.jpg          – symlinks → existing {:04d}.png  (viz_overlay expects 6-digit jpg)

Usage:
  python convert_sam3d.py data/bowl6
  python convert_sam3d.py data/bowl6 --pose-R R_cum --pose-t t
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np


# ── poses ─────────────────────────────────────────────────────────────────────

def convert_poses(data_dir: Path, pose_R: str, pose_t: str) -> None:
    d = np.load(data_dir / "poses.npz")
    R = d[pose_R].astype(np.float64)   # (N, 3, 3)
    t = d[pose_t].astype(np.float64)   # (N, 3)
    N = len(R)

    poses = np.tile(np.eye(4), (N, 1, 1))
    poses[:, :3, :3] = R
    poses[:, :3,  3] = t
    frames = np.arange(N, dtype=np.int64)

    dst_dir = data_dir / "object_pose"
    dst_dir.mkdir(exist_ok=True)
    np.savez(dst_dir / "poses.npz", poses=poses, frames=frames)
    print(f"  poses  → {dst_dir / 'poses.npz'}  ({N} frames, keys: R={pose_R}, t={pose_t})")


# ── camera ────────────────────────────────────────────────────────────────────

def convert_camera(data_dir: Path) -> None:
    meta = json.loads((data_dir / "meta.json").read_text())
    intr = meta["intrinsics"]   # {fx, fy, cx, cy}
    h, w = meta["image_size"]   # [H, W]

    camera = {
        "height": h,
        "width":  w,
        "intrinsics": [
            [intr["fx"], 0.0,        intr["cx"]],
            [0.0,        intr["fy"], intr["cy"]],
            [0.0,        0.0,        1.0        ],
        ],
    }
    dst = data_dir / "camera.json"
    dst.write_text(json.dumps(camera, indent=2))
    print(f"  camera → {dst}  ({w}x{h})")


# ── mesh ──────────────────────────────────────────────────────────────────────

def convert_mesh(data_dir: Path) -> None:
    try:
        import trimesh
    except ImportError:
        raise SystemExit(
            "trimesh is required for mesh conversion.\n"
            "Install with:  pip install trimesh[easy]"
        )
    from PIL import Image

    meta  = json.loads((data_dir / "meta.json").read_text())
    scale = float(meta["sam3d_scale_to_meters"])

    # GLB GLTF Y-up → OpenCV camera-local frame
    # meta.json recipe: v_cv_local = stack(-x_glb, z_glb, y_glb) * scale
    M = scale * np.array([[-1., 0., 0.],
                           [ 0., 0., 1.],
                           [ 0., 1., 0.]])

    scene = trimesh.load(str(data_dir / "sam3d_mesh.glb"), force="scene")
    if isinstance(scene, trimesh.Scene):
        parts = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh  = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
    else:
        mesh = scene

    mesh.vertices = (M @ mesh.vertices.T).T

    mesh_dir = data_dir / "mesh"
    mesh_dir.mkdir(exist_ok=True)

    obj_path = mesh_dir / "textured_simple.obj"
    # trimesh may write vertex colors as "v x y z r g b"; prepare_mesh() strips them
    result = trimesh.exchange.obj.export_obj(mesh)
    if isinstance(result, str):
        obj_path.write_text(result)
    else:
        obj_path.write_bytes(result)
    print(f"  mesh   → {obj_path}  ({len(mesh.vertices)} verts, {len(mesh.faces)} faces)")

    tex_path = mesh_dir / "texture_map.png"
    if not tex_path.exists():
        # Flat gray placeholder — MuJoCo needs a texture file even for solid-color rendering
        img = Image.fromarray(np.full((4, 4, 3), 160, dtype=np.uint8))
        img.save(str(tex_path))
        print(f"  tex    → {tex_path}  (placeholder)")
    else:
        print(f"  tex    → {tex_path}  (already exists, kept)")


# ── rgb symlinks ──────────────────────────────────────────────────────────────

def convert_rgb(data_dir: Path) -> None:
    rgb_dir = data_dir / "rgb"
    pngs    = sorted(rgb_dir.glob("????.png"))
    if not pngs:
        print("  rgb    → no {:04d}.png files found, skipping")
        return

    created = 0
    for src in pngs:
        frame_idx = int(src.stem)
        dst = rgb_dir / f"{frame_idx:06d}.jpg"
        if not dst.exists():
            os.symlink(src.name, dst)   # relative symlink inside rgb/
            created += 1

    print(f"  rgb    → {created} symlinks NNNNNN.jpg → NNNN.png  ({len(pngs)} frames)")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SAM3D data dir to main_abs.py / viz_overlay.py format"
    )
    parser.add_argument("data_dir", help="Path to SAM3D data directory (e.g. data/bowl6)")
    parser.add_argument(
        "--pose-R", default="R_sm",
        choices=["R_sm", "R_cum", "R_sm_anc", "R_cum_anc"],
        help="Rotation key in poses.npz (default: R_sm)",
    )
    parser.add_argument(
        "--pose-t", default="t_sm",
        choices=["t_sm", "t"],
        help="Translation key in poses.npz (default: t_sm)",
    )
    parser.add_argument(
        "--skip-mesh", action="store_true",
        help="Skip GLB→OBJ conversion (requires trimesh)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        # Resolve relative to project root (two levels above this script: simulation/ → project/)
        data_dir = Path(__file__).resolve().parents[1] / data_dir

    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    print(f"Converting {data_dir} ...")
    convert_poses(data_dir, args.pose_R, args.pose_t)
    convert_camera(data_dir)
    if not args.skip_mesh:
        convert_mesh(data_dir)
    convert_rgb(data_dir)
    print("Done.")


if __name__ == "__main__":
    main()
