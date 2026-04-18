import cv2
import numpy as np


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
        [1,2,3], [(0,0,255),(0,255,0),(255,0,0)], ["+X","+Y","+Z"]
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
