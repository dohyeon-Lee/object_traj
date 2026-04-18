# Usage:
#   python src/visualize_dataset.py
#   python src/visualize_dataset.py data/006_mustard_bottle_20200709_143211
#   python src/visualize_dataset.py data/006_mustard_bottle_20200709_143211 --fps 20
#
# Output: videos/<dataset_folder_name>/trajectory.mp4

import json
import os
from pathlib import Path

import cv2
import imageio
import numpy as np

os.environ.setdefault("MUJOCO_GL", "osmesa")


def load_data(data_dir):
    data_dir = Path(data_dir)
    npz = np.load(data_dir / "object_pose" / "poses.npz")
    poses = npz["poses"]    # (N, 4, 4)
    frames = npz["frames"]  # (N,) frame indices

    cam = json.loads((data_dir / "camera.json").read_text())
    K = np.array(cam["intrinsics"])
    E = np.array(cam["extrinsics"])

    rgb_dir = data_dir / "rgb"
    images = []
    for f in frames:
        img_path = rgb_dir / f"{f:06d}.jpg"
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)

    return poses, frames, images, K, E


def project_points(pts_3d, K):
    pts_h = pts_3d / pts_3d[:, 2:3]
    px = (K @ pts_h.T).T
    return px[:, :2]


def draw_axes(img, pose_cam, K, length=0.05):
    origin = pose_cam[:3, 3]
    if origin[2] < 0.01:
        return img

    axes = np.array([
        origin,
        origin + pose_cam[:3, 0] * length,
        origin + pose_cam[:3, 1] * length,
        origin + pose_cam[:3, 2] * length,
    ])
    px = project_points(axes, K).astype(int)
    H, W = img.shape[:2]

    def in_bounds(p):
        return 0 <= p[0] < W and 0 <= p[1] < H

    for j, c in enumerate([(255, 50, 50), (50, 255, 50), (50, 50, 255)]):
        if in_bounds(px[0]) and in_bounds(px[j + 1]):
            cv2.arrowedLine(img, tuple(px[0]), tuple(px[j + 1]), c, 2,
                            tipLength=0.3, line_type=cv2.LINE_AA)
    return img


def draw_trajectory(images, poses_cam, K, color, radius=6, trail_len=None):
    origins = poses_cam[:, :3, 3]
    valid = origins[:, 2] > 0.01
    pixels = np.full((len(origins), 2), -1, dtype=float)
    pixels[valid] = project_points(origins[valid], K)

    out = []
    for i, img in enumerate(images):
        frame = img.copy()
        H, W = frame.shape[:2]

        start = 0 if trail_len is None else max(0, i - trail_len)
        trail_pts = []
        for j in range(start, i + 1):
            u, v = pixels[j]
            if 0 <= u < W and 0 <= v < H:
                trail_pts.append((int(u), int(v)))

        for k in range(1, len(trail_pts)):
            alpha = k / len(trail_pts)
            c = tuple(int(x * alpha) for x in color)
            cv2.line(frame, trail_pts[k - 1], trail_pts[k], c, 2, cv2.LINE_AA)

        u, v = pixels[i]
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(frame, (int(u), int(v)), radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (int(u), int(v)), radius + 2, (255, 255, 255), 1, cv2.LINE_AA)

        frame = draw_axes(frame, poses_cam[i], K, length=0.05)
        out.append(frame)
    return out


def make_video(data_dir, fps=10):
    data_dir = Path(data_dir)
    out_dir = Path("videos") / data_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    poses, frames, images, K, E = load_data(data_dir)
    print(f"Loaded {len(poses)} frames")

    frames_drawn = draw_trajectory(images, poses, K, color=(0, 200, 255), trail_len=20)

    path = str(out_dir / "trajectory.mp4")
    imageio.mimwrite(path, frames_drawn, fps=fps, codec="libx264",
                     output_params=["-crf", "18"])
    print(f"Saved -> {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", nargs="?",
                        default="data/006_mustard_bottle_20200709_143211")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    make_video(args.data_dir, fps=args.fps)
