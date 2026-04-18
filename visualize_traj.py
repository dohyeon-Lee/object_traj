"""
Visualize object trajectory projected onto RGB images.

Tests two conventions for camera.json extrinsics:
  - "cam2world": poses.npz is in camera frame (project directly)
  - "world2cam": poses.npz is in world frame (apply inv(extrinsics) first)

Saves two side-by-side videos so you can see which one looks correct.
"""

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
    K = np.array(cam["intrinsics"])          # (3, 3)
    E = np.array(cam["extrinsics"])          # (4, 4)

    # load rgb images matching frame indices
    rgb_dir = data_dir / "rgb"
    images = []
    for f in frames:
        img_path = rgb_dir / f"{f:06d}.jpg"
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)

    return poses, frames, images, K, E


def project_points(pts_3d, K):
    """
    pts_3d: (N, 3) in camera frame
    K:      (3, 3) intrinsics
    returns: (N, 2) pixel coords
    """
    pts_h = pts_3d / pts_3d[:, 2:3]         # normalize by Z
    px = (K @ pts_h.T).T                     # (N, 3)
    return px[:, :2]                         # (N, 2) uv


def draw_trajectory(images, poses_cam, K, color, radius=6, trail_len=None):
    """
    poses_cam: (N, 4, 4) object poses in camera frame
    Draws the object origin projected onto each frame, plus a trailing path.
    """
    origins = poses_cam[:, :3, 3]            # (N, 3) translation in cam frame

    # filter out poses behind camera
    valid = origins[:, 2] > 0.01

    pixels = np.full((len(origins), 2), -1, dtype=float)
    pixels[valid] = project_points(origins[valid], K)

    out = []
    for i, img in enumerate(images):
        frame = img.copy()
        H, W = frame.shape[:2]

        # draw trail
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

        # draw current position
        u, v = pixels[i]
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(frame, (int(u), int(v)), radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (int(u), int(v)), radius + 2, (255, 255, 255), 1, cv2.LINE_AA)

        # draw axes (x=red, y=green, z=blue)
        frame = draw_axes(frame, poses_cam[i], K, length=0.05)

        out.append(frame)
    return out


def draw_axes(img, pose_cam, K, length=0.05):
    """Draw XYZ axes of the object frame projected onto image."""
    origin = pose_cam[:3, 3]
    if origin[2] < 0.01:
        return img

    axes = np.array([
        origin,
        origin + pose_cam[:3, 0] * length,  # X red
        origin + pose_cam[:3, 1] * length,  # Y green
        origin + pose_cam[:3, 2] * length,  # Z blue
    ])
    px = project_points(axes, K).astype(int)
    H, W = img.shape[:2]

    def in_bounds(p):
        return 0 <= p[0] < W and 0 <= p[1] < H

    colors_axes = [(255, 50, 50), (50, 255, 50), (50, 50, 255)]
    for j, c in enumerate(colors_axes):
        if in_bounds(px[0]) and in_bounds(px[j + 1]):
            cv2.arrowedLine(img, tuple(px[0]), tuple(px[j + 1]), c, 2,
                            tipLength=0.3, line_type=cv2.LINE_AA)
    return img


def add_label(img, text):
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def make_video(data_dir, out_dir="videos/verify", fps=10):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    poses, frames, images, K, E = load_data(data_dir)
    N = len(poses)
    print(f"Loaded {N} frames | K:\n{K.round(1)}")
    print(f"Extrinsics:\n{E.round(4)}")

    # ── interpretation A: poses already in camera frame ──────────────────────
    poses_camA = poses  # use directly

    # ── interpretation B: poses in world frame, convert to camera frame ───────
    # T_world2cam = E  →  T_cam_obj = T_world2cam @ T_world_obj
    T_world2cam = E
    poses_camB = T_world2cam @ poses   # (N, 4, 4)

    # also try B with inverse (in case E is cam2world)
    T_cam2world = E
    T_world2cam_inv = np.linalg.inv(T_cam2world)
    poses_camC = T_world2cam_inv @ poses

    frames_drawn = draw_trajectory(images, poses_camA, K, color=(0, 200, 255), trail_len=20)
    for f in frames_drawn:
        add_label(f, "poses in camera frame")

    path = str(out_dir / "trajectory.mp4")
    imageio.mimwrite(path, frames_drawn, fps=fps, codec="libx264",
                     output_params=["-crf", "18"])
    print(f"Saved -> {path}")


def visualize_3d(data_dir, out_dir="videos/verify"):
    """Interactive 3D trajectory plot (HTML) — open in browser to rotate."""
    import plotly.graph_objects as go

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    poses, frames, _, K, _ = load_data(data_dir)
    pos = poses[:, :3, 3]   # (N, 3) in camera frame
    N = len(pos)
    t = np.arange(N)

    # equal axis range
    all_pts = np.vstack([pos, [[0, 0, 0]]])
    mid = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
    half = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.1

    fig = go.Figure()

    # trajectory
    fig.add_trace(go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        mode="lines+markers",
        marker=dict(size=4, color=t, colorscale="Plasma", showscale=True,
                    colorbar=dict(title="frame")),
        line=dict(color="gray", width=2),
        name="trajectory",
    ))

    # start / end
    fig.add_trace(go.Scatter3d(
        x=[pos[0, 0]], y=[pos[0, 1]], z=[pos[0, 2]],
        mode="markers", marker=dict(size=8, color="lime", symbol="diamond"),
        name="start",
    ))
    fig.add_trace(go.Scatter3d(
        x=[pos[-1, 0]], y=[pos[-1, 1]], z=[pos[-1, 2]],
        mode="markers", marker=dict(size=8, color="red", symbol="diamond"),
        name="end",
    ))

    # camera at origin
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers", marker=dict(size=8, color="cyan", symbol="square"),
        name="camera",
    ))

    fig.update_layout(
        title="Object trajectory in camera frame (drag to rotate)",
        scene=dict(
            xaxis=dict(title="X (m)", range=[mid[0]-half, mid[0]+half]),
            yaxis=dict(title="Y (m)", range=[mid[1]-half, mid[1]+half]),
            zaxis=dict(title="Z (m)", range=[mid[2]-half, mid[2]+half]),
            aspectmode="cube",
        ),
    )

    out_path = str(out_dir / "traj_3d.html")
    fig.write_html(out_path)
    print(f"Saved -> {out_path}  (open in browser)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", nargs="?",
                        default="006_mustard_bottle_20200709_143211")
    parser.add_argument("--out-dir", default="videos/verify")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--3d", action="store_true", dest="plot3d",
                        help="save 3D trajectory plot instead of video")
    args = parser.parse_args()

    if args.plot3d:
        visualize_3d(args.data_dir, out_dir=args.out_dir)
    else:
        make_video(args.data_dir, out_dir=args.out_dir, fps=args.fps)
