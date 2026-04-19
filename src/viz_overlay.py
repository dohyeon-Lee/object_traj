import json
import cv2
import imageio
import numpy as np
from pathlib import Path


def get_cam_matrices(env, camera_name, height, width):
    cam_id  = env.sim.model.camera_name2id(camera_name)
    fovy    = env.sim.model.cam_fovy[cam_id]
    f       = height / (2 * np.tan(np.deg2rad(fovy) / 2))
    K       = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])
    cam_pos = env.sim.data.cam_xpos[cam_id].copy()
    cam_rot = env.sim.data.cam_xmat[cam_id].reshape(3, 3).copy()
    return K, cam_pos, cam_rot


def project(points_world, K, cam_pos, cam_rot, img_height):
    """(N,3) world → (N,2) pixel, accounting for robosuite's vertical image flip."""
    pts = (cam_rot.T @ (np.array(points_world) - cam_pos).T).T
    pts[:, 1] *= -1   # MuJoCo Y-up → image Y-down
    pts[:, 2] *= -1   # MuJoCo Z-backward → Z-forward
    in_front = pts[:, 2] > 0
    px = np.full((len(pts), 2), -1.0)
    px[in_front] = (K @ pts[in_front].T).T[:, :2] / pts[in_front, 2:3]
    return px, in_front


def draw_axes(img, K, cam_pos, cam_rot, origin, length=0.1, name=None):
    """Draw XYZ axes at a world-frame origin onto img (in-place)."""
    H, W = img.shape[:2]
    o = np.array(origin, dtype=float)
    pts = np.array([o, o + [length,0,0], o + [0,length,0], o + [0,0,length]])
    px, valid = project(pts, K, cam_pos, cam_rot, H)

    def ok(p):
        return 0 <= p[0] < W and 0 <= p[1] < H

    if not (valid[0] and ok(px[0])):
        return img

    p0 = tuple(px[0].astype(int))
    if name:
        cv2.putText(img, name, (p0[0]+4, p0[1]-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(img, name, (p0[0]+4, p0[1]-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)

    for axis, color, label in zip(
        [1,2,3], [(255,0,0),(0,255,0),(0,0,255)], ["+X","+Y","+Z"]
    ):
        if valid[axis] and ok(px[axis]):
            p1 = tuple(px[axis].astype(int))
            cv2.arrowedLine(img, p0, p1, color, 2, tipLength=0.25, line_type=cv2.LINE_AA)
            # offset label in the arrow direction so it doesn't overlap the tip
            direction = px[axis] - px[0]
            norm = np.linalg.norm(direction)
            offset = (direction / norm * 14).astype(int) if norm > 0 else np.array([8, -8])
            lpos = tuple((px[axis].astype(int) + offset))
            cv2.putText(img, label, lpos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(img, label, lpos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return img


def draw_eef(img, eef_positions, eef_quat, K, cam_pos, cam_rot,
             trail_len=40, dot_radius=6, axis_length=0.05):
    """Draw EEF trail, current dot, and orientation axes onto img (in-place).

    eef_positions: list of (3,) world-frame positions (history so far)
    eef_quat:      (4,) xyzw quaternion of current EEF orientation
    """
    from scipy.spatial.transform import Rotation
    H, W = img.shape[:2]

    def ok(p):
        return 0 <= p[0] < W and 0 <= p[1] < H

    # ── trail + dot ───────────────────────────────────────────────────────────
    trail = eef_positions[-trail_len:]
    px_trail, valid_trail = project(trail, K, cam_pos, cam_rot, H)

    for k in range(1, len(trail)):
        if valid_trail[k - 1] and valid_trail[k]:
            p0 = tuple(px_trail[k - 1].astype(int))
            p1 = tuple(px_trail[k].astype(int))
            if ok(p0) and ok(p1):
                alpha = k / len(trail)
                color = tuple(int(c * alpha) for c in (0, 220, 255))
                cv2.line(img, p0, p1, color, 2, cv2.LINE_AA)

    if valid_trail[-1]:
        dot = tuple(px_trail[-1].astype(int))
        if ok(dot):
            cv2.circle(img, dot, dot_radius,     (0, 220, 255), -1, cv2.LINE_AA)
            cv2.circle(img, dot, dot_radius + 2, (255, 255, 255),  1, cv2.LINE_AA)

    # ── orientation axes ──────────────────────────────────────────────────────
    eef_pos = np.array(eef_positions[-1])
    R_eef   = Rotation.from_quat(eef_quat).as_matrix()

    axis_pts = np.array([eef_pos] + [eef_pos + R_eef[:, i] * axis_length for i in range(3)])
    px_ax, valid_ax = project(axis_pts, K, cam_pos, cam_rot, H)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    labels = ["+X", "+Y", "+Z"]
    if valid_ax[0] and ok(tuple(px_ax[0].astype(int))):
        p0 = tuple(px_ax[0].astype(int))
        for i, (color, label) in enumerate(zip(colors, labels)):
            if valid_ax[i + 1]:
                p1 = tuple(px_ax[i + 1].astype(int))
                if ok(p1):
                    cv2.arrowedLine(img, p0, p1, color, 2, tipLength=0.25, line_type=cv2.LINE_AA)
                    direction = px_ax[i + 1] - px_ax[0]
                    norm = np.linalg.norm(direction)
                    offset = (direction / norm * 14).astype(int) if norm > 0 else np.array([8, -8])
                    lpos = tuple(px_ax[i + 1].astype(int) + offset)
                    cv2.putText(img, label, lpos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(img, label, lpos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color,            1, cv2.LINE_AA)

    return img


# ── dataset trajectory video ──────────────────────────────────────────────────

def _load_data(data_dir):
    data_dir = Path(data_dir)
    npz    = np.load(data_dir / "object_pose" / "poses.npz")
    poses  = npz["poses"]
    frames = npz["frames"]
    cam    = json.loads((data_dir / "camera.json").read_text())
    K      = np.array(cam["intrinsics"])
    images = []
    for f in frames:
        img = cv2.imread(str(data_dir / "rgb" / f"{f:06d}.jpg"))
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return poses, images, K


def _project_pts(pts_3d, K):
    pts_h = pts_3d / pts_3d[:, 2:3]
    return (K @ pts_h.T).T[:, :2]


def _draw_object_axes(img, pose_cam, K, length=0.05):
    origin = pose_cam[:3, 3]
    if origin[2] < 0.01:
        return img
    pts = np.array([origin,
                    origin + pose_cam[:3, 0] * length,
                    origin + pose_cam[:3, 1] * length,
                    origin + pose_cam[:3, 2] * length])
    px  = _project_pts(pts, K).astype(int)
    H, W = img.shape[:2]
    def ok(p): return 0 <= p[0] < W and 0 <= p[1] < H
    for j, c in enumerate([(255, 50, 50), (50, 255, 50), (50, 50, 255)]):
        if ok(px[0]) and ok(px[j + 1]):
            cv2.arrowedLine(img, tuple(px[0]), tuple(px[j + 1]), c, 2,
                            tipLength=0.3, line_type=cv2.LINE_AA)
    return img


def _draw_trajectory(images, poses_cam, K, color, radius=6, trail_len=20):
    origins = poses_cam[:, :3, 3]
    valid   = origins[:, 2] > 0.01
    pixels  = np.full((len(origins), 2), -1, dtype=float)
    pixels[valid] = _project_pts(origins[valid], K)
    out = []
    for i, img in enumerate(images):
        frame = img.copy()
        H, W  = frame.shape[:2]
        start = max(0, i - trail_len)
        trail = [(int(pixels[j, 0]), int(pixels[j, 1]))
                 for j in range(start, i + 1)
                 if 0 <= pixels[j, 0] < W and 0 <= pixels[j, 1] < H]
        for k in range(1, len(trail)):
            alpha = k / len(trail)
            c = tuple(int(x * alpha) for x in color)
            cv2.line(frame, trail[k - 1], trail[k], c, 2, cv2.LINE_AA)
        u, v = pixels[i]
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(frame, (int(u), int(v)), radius,     color,           -1, cv2.LINE_AA)
            cv2.circle(frame, (int(u), int(v)), radius + 2, (255, 255, 255),  1, cv2.LINE_AA)
        frame = _draw_object_axes(frame, poses_cam[i], K)
        out.append(frame)
    return out


def make_video(data_dir, fps=10):
    """Generate trajectory.mp4 from dataset RGB frames + pose data."""
    data_dir = Path(data_dir)
    out_dir  = data_dir.parent.parent / "videos" / data_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    poses, images, K = _load_data(data_dir)
    print(f"[visualize] Loaded {len(poses)} frames from {data_dir.name}")
    frames_drawn = _draw_trajectory(images, poses, K, color=(0, 200, 255))
    path = str(out_dir / "trajectory.mp4")
    imageio.mimwrite(path, frames_drawn, fps=fps, codec="libx264",
                     output_params=["-crf", "18"])
    print(f"[visualize] Saved -> {path}")
