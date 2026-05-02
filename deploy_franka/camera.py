"""
Camera calibration using wrist camera + ChArUco board.

Steps:
  1) calibrate_wrist  – hand-eye calibration for wrist camera (one-time)
  2) locate_board     – find board pose in robot frame using wrist cam
  3) track            – semi-real-time 3rd person camera pose tracking

Usage:
  python camera.py calibrate_wrist   # move robot to ~10 poses, press Enter each time
  python camera.py locate_board      # single shot, saves board pose
  python camera.py track             # live 3rd person camera pose
  python camera.py align --data-dir data/035_power_drill_20200709_151335 --angle 90 --scale 1.0

"""

import argparse
import json
import pickle
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

CALIB_DIR = Path(__file__).parent / "calibration"
WRIST_CAM_FILE = CALIB_DIR / "wrist_cam_offset.json"
BOARD_POSE_FILE = CALIB_DIR / "board_pose.json"

_WORKER_SCRIPT = Path(__file__).parent / "_aruco_worker.py"
_WORKER_PYTHON = str(Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python")
_worker_proc = None


def _get_worker():
    """Start (or reuse) the aruco subprocess worker (uses .venv Python with OpenCV 4.13)."""
    global _worker_proc
    if _worker_proc is not None and _worker_proc.poll() is None:
        return _worker_proc
    _worker_proc = subprocess.Popen(
        [_WORKER_PYTHON, str(_WORKER_SCRIPT)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return _worker_proc


def _worker_call(image, K, dist):
    """Send image to worker, get detection + pose result."""
    proc = _get_worker()
    msg = pickle.dumps({
        "image": image.tobytes(), "image_shape": image.shape, "image_dtype": str(image.dtype),
        "K": K.tolist(), "dist": dist.tolist(),
    }, protocol=2)
    proc.stdin.write(struct.pack(">I", len(msg)))
    proc.stdin.write(msg)
    proc.stdin.flush()
    raw = proc.stdout.read(4)
    if len(raw) < 4:
        rc = proc.poll()
        stderr_out = ""
        if proc.stderr:
            stderr_out = proc.stderr.read().decode(errors="replace")
        print(f"  [worker] died (rc={rc}): {stderr_out[:200]}")
        return None
    size = struct.unpack(">I", raw)[0]
    data = proc.stdout.read(size)
    return pickle.loads(data)


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_env():
    from droid.robot_env import RobotEnv
    return RobotEnv(
        action_space="cartesian_position",
        gripper_action_space="position",
        camera_kwargs=dict(
            hand_camera=dict(image=True, concatenate_images=False, resize_func="cv2"),
            varied_camera=dict(image=True, concatenate_images=False, resize_func="cv2"),
        ),
    )


def _get_intrinsics(obs, cam_key):
    """Return (camera_matrix, dist_coeffs) for a camera."""
    K = obs["camera_intrinsics"][cam_key]
    if isinstance(K, list):
        K = np.array(K)
    return K.astype(np.float64), np.zeros(5)


def _detect_and_pose(image, camera_matrix, dist_coeffs):
    """Detect ChArUco + estimate pose via subprocess. Returns dict or None."""
    result = _worker_call(image, camera_matrix, dist_coeffs)
    if result is not None and result["rvec"] is not None:
        result["rvec"] = np.array(result["rvec"])
        result["tvec"] = np.array(result["tvec"])
    return result


def _flange_pose(obs):
    """Return (pos, rotmat) of flange from robot state."""
    state = obs["robot_state"]
    pos = np.array(state["cartesian_position"][:3])
    rotvec = np.array(state["cartesian_position"][3:])
    R = Rotation.from_rotvec(rotvec).as_matrix()
    return pos, R


def _pose_to_4x4(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved → {path}")


def _load_json(path):
    with open(path) as f:
        return json.load(f)


# ── robot movement helper ────────────────────────────────────────────────────

def _move_to(env, pose6, steps=80, hz=10):
    """Move robot to a cartesian pose [x,y,z,rx,ry,rz], gripper open."""
    action = np.concatenate([np.array(pose6, dtype=np.float32), [0.0]])
    for _ in range(steps):
        env.step(action)
        time.sleep(1.0 / hz)


def _generate_calib_poses(center_pos, center_rotvec, n=12):
    """Generate diverse poses around current position for hand-eye calibration."""
    poses = []
    pos_offsets = [
        [ 0.00,  0.00,  0.00],
        [ 0.05,  0.00,  0.00],
        [-0.05,  0.00,  0.00],
        [ 0.00,  0.05,  0.00],
        [ 0.00, -0.05,  0.00],
        [ 0.00,  0.00,  0.03],
        [ 0.00,  0.00, -0.03],
        [ 0.04,  0.04,  0.00],
        [-0.04,  0.04,  0.00],
        [ 0.04, -0.04,  0.00],
        [-0.04, -0.04,  0.00],
        [ 0.03,  0.03,  0.03],
    ]
    rot_offsets_deg = [
        [ 0,  0,  0],
        [10,  0,  0],
        [-10, 0,  0],
        [ 0, 10,  0],
        [ 0,-10,  0],
        [ 0,  0, 10],
        [ 0,  0,-10],
        [ 8,  8,  0],
        [-8,  8,  0],
        [ 8, -8,  0],
        [-8, -8,  0],
        [ 5,  5,  5],
    ]
    R0 = Rotation.from_rotvec(center_rotvec)
    for i in range(min(n, len(pos_offsets))):
        p = center_pos + np.array(pos_offsets[i])
        dR = Rotation.from_euler('xyz', rot_offsets_deg[i], degrees=True)
        rv = (dR * R0).as_rotvec()
        poses.append(np.concatenate([p, rv]))
    return poses


# ── Step 1: Calibrate wrist camera (hand-eye) ───────────────────────────────

def calibrate_wrist(env, n_poses=12):
    """
    Automatic hand-eye calibration: robot moves to diverse poses around
    the current position while capturing ChArUco board detections.
    """
    print("=== Wrist Camera Hand-Eye Calibration (auto) ===")
    print("Place ChArUco board on the table, visible to wrist camera.")
    print("The robot will automatically move to different poses.\n")

    obs = env.get_observation()
    state = obs["robot_state"]["cartesian_position"]
    center_pos = np.array(state[:3])
    center_rv = np.array(state[3:])

    # find camera keys
    hand_cam_key = None
    intr_key = None
    image_dict = obs.get("image", {})
    for key in image_dict:
        if "11049903" in key:
            hand_cam_key = key
            break
    if hand_cam_key is None:
        for key in image_dict:
            if "_left" in key:
                hand_cam_key = key
                break
    for k in obs.get("camera_intrinsics", {}):
        if "hand" in k or "11049903" in k:
            intr_key = k
            break
    if hand_cam_key is None or intr_key is None:
        print("Cannot find hand camera. image keys:", list(image_dict.keys()))
        return
    print(f"Using camera: image[{hand_cam_key}], intrinsics: {intr_key}")

    poses = _generate_calib_poses(center_pos, center_rv, n_poses)
    print(f"Will visit {len(poses)} poses around current position.")
    input("Press Enter to start (make sure board is visible and area is clear)...")

    R_gripper2base_list = []
    t_gripper2base_list = []
    R_target2cam_list = []
    t_target2cam_list = []
    count = 0

    for i, target_pose in enumerate(poses):
        print(f"\n[{i+1}/{len(poses)}] Moving to pose...")
        _move_to(env, target_pose.tolist())
        time.sleep(0.5)

        obs = env.get_observation()
        image = obs["image"].get(hand_cam_key)
        if image is None:
            print("  No image, skipping")
            continue

        K, dist = _get_intrinsics(obs, intr_key)
        result = _detect_and_pose(image, K, dist)
        if result is None:
            print("  Board not detected, skipping")
            continue
        if result["rvec"] is None:
            print("  Pose estimation failed, skipping")
            continue

        rvec_board, tvec_board = result["rvec"], result["tvec"]
        R_board = Rotation.from_rotvec(rvec_board).as_matrix()
        flange_pos, flange_R = _flange_pose(obs)

        R_gripper2base_list.append(flange_R)
        t_gripper2base_list.append(flange_pos)
        R_target2cam_list.append(R_board)
        t_target2cam_list.append(tvec_board)

        count += 1
        print(f"  Captured sample {count} (board at {tvec_board.round(3)})")

    # return to start
    print("\nReturning to start pose...")
    _move_to(env, np.concatenate([center_pos, center_rv]).tolist())

    if count < 3:
        print(f"Only {count} samples captured. Need at least 3.")
        return

    import cv2
    print(f"\nRunning hand-eye calibration with {count} samples...")
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base_list,
        t_gripper2base=t_gripper2base_list,
        R_target2cam=R_target2cam_list,
        t_target2cam=t_target2cam_list,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )

    t_cam2gripper = t_cam2gripper.flatten()
    rotvec = Rotation.from_matrix(R_cam2gripper).as_rotvec()

    print(f"Camera-to-gripper translation: {t_cam2gripper.round(4)}")
    print(f"Camera-to-gripper rotation (rotvec): {rotvec.round(4)}")

    _save_json(WRIST_CAM_FILE, {
        "R": R_cam2gripper.tolist(),
        "t": t_cam2gripper.tolist(),
    })


# ── Step 2: Locate board in robot frame ──────────────────────────────────────

def locate_board(env):
    """
    Use wrist camera + calibrated offset to find board pose in robot frame.
    """
    if not WRIST_CAM_FILE.exists():
        print(f"Run 'calibrate_wrist' first. Missing: {WRIST_CAM_FILE}")
        return

    calib = _load_json(WRIST_CAM_FILE)
    R_cam2gripper = np.array(calib["R"])
    t_cam2gripper = np.array(calib["t"])
    T_cam2gripper = _pose_to_4x4(R_cam2gripper, t_cam2gripper)

    print("=== Locate Board ===")
    print("Move wrist camera so it can see the board, then press Enter.")
    input()

    obs = env.get_observation()

    # find hand camera inside obs['image'] dict
    hand_cam_key = None
    intr_key = None
    image_dict = obs.get("image", {})
    for key in image_dict:
        if "11049903" in key and "_left" in key:
            hand_cam_key = key
            break
    if hand_cam_key is None:
        for key in image_dict:
            if "11049903" in key:
                hand_cam_key = key
                break
    for k in obs.get("camera_intrinsics", {}):
        if "11049903" in k:
            intr_key = k

    if hand_cam_key is None or intr_key is None:
        print("Cannot find hand camera. image keys:", list(image_dict.keys()))
        return

    image = obs["image"][hand_cam_key]
    K, dist = _get_intrinsics(obs, intr_key)
    result = _detect_and_pose(image, K, dist)
    if result is None:
        print("Board not detected")
        return
    if result["rvec"] is None:
        print("Pose estimation failed")
        return

    rvec_board, tvec_board = result["rvec"], result["tvec"]
    R_board2cam = Rotation.from_rotvec(rvec_board).as_matrix()
    T_board2cam = _pose_to_4x4(R_board2cam, tvec_board)

    flange_pos, flange_R = _flange_pose(obs)
    T_gripper2base = _pose_to_4x4(flange_R, flange_pos)

    # board_in_robot = gripper_in_robot @ cam_in_gripper @ board_in_cam
    T_board2base = T_gripper2base @ T_cam2gripper @ T_board2cam

    pos = T_board2base[:3, 3]
    rotvec = Rotation.from_matrix(T_board2base[:3, :3]).as_rotvec()
    print(f"Board position in robot frame: {pos.round(4)}")
    print(f"Board orientation (rotvec): {rotvec.round(4)}")

    _save_json(BOARD_POSE_FILE, {
        "R": T_board2base[:3, :3].tolist(),
        "t": T_board2base[:3, 3].tolist(),
    })


# ── Step 3: Track 3rd person camera ─────────────────────────────────────────

def track(env):
    """
    Semi-real-time 3rd person camera pose tracking.
    Board position must be known (run locate_board first).
    """
    if not BOARD_POSE_FILE.exists():
        print(f"Run 'locate_board' first. Missing: {BOARD_POSE_FILE}")
        return

    board = _load_json(BOARD_POSE_FILE)
    R_board2base = np.array(board["R"])
    t_board2base = np.array(board["t"])
    T_board2base = _pose_to_4x4(R_board2base, t_board2base)

    from PIL import Image
    track_img_path = Path(__file__).parent / "track_view.png"
    _last_save = 0.0
    _SAVE_INTERVAL = 1.0

    print("=== 3rd Person Camera Tracking ===")
    print(f"Live image saved to: {track_img_path} (every {_SAVE_INTERVAL}s)")
    print("Press Ctrl+C to stop.\n")

    varied_cam_key = None
    intr_key = None

    try:
        while True:
            obs = env.get_observation()

            # find varied camera inside obs['image'] dict
            if varied_cam_key is None:
                image_dict = obs.get("image", {})
                for key in image_dict:
                    if "36776608" in key and "_left" in key:
                        varied_cam_key = key
                        break
                if varied_cam_key is None:
                    for key in image_dict:
                        if "36776608" in key:
                            varied_cam_key = key
                            break
                for k in obs.get("camera_intrinsics", {}):
                    if "36776608" in k:
                        intr_key = k
                if varied_cam_key is None or intr_key is None:
                    print("Cannot find varied camera. image keys:", list(image_dict.keys()))
                    return
                print(f"Using camera: image[{varied_cam_key}], intrinsics: {intr_key}")

            image = obs["image"][varied_cam_key]

            # save live view at reduced rate
            now = time.time()
            if now - _last_save >= _SAVE_INTERVAL:
                Image.fromarray(image).save(track_img_path)
                _last_save = now

            K, dist = _get_intrinsics(obs, intr_key)
            result = _detect_and_pose(image, K, dist)

            if result is None:
                print("Board not visible")
                time.sleep(0.1)
                continue
            if result["rvec"] is None:
                print("Pose estimation failed")
                time.sleep(0.1)
                continue

            rvec_board, tvec_board = result["rvec"], result["tvec"]
            R_board2cam = Rotation.from_rotvec(rvec_board).as_matrix()
            T_board2cam = _pose_to_4x4(R_board2cam, tvec_board)

            # cam_in_robot = board_in_robot @ inv(board_in_cam)
            T_cam2base = T_board2base @ np.linalg.inv(T_board2cam)

            cam_pos = T_cam2base[:3, 3]
            cam_rotvec = Rotation.from_matrix(T_cam2base[:3, :3]).as_rotvec()
            print(f"Camera pos: [{cam_pos[0]:+.4f}, {cam_pos[1]:+.4f}, {cam_pos[2]:+.4f}]  "
                  f"rot: [{cam_rotvec[0]:+.3f}, {cam_rotvec[1]:+.3f}, {cam_rotvec[2]:+.3f}]")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopped.")


# ── Step 4: Align real camera to simulation camera ────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_CAM_FRAME_IN_ROBOT = np.array([[0, 0, -1],
                                        [1, 0,  0],
                                        [0, -1, 0]], dtype=float)


def _cam_to_robot_matrix(angle_deg):
    return Rotation.from_euler('z', -angle_deg, degrees=True).as_matrix() @ DATASET_CAM_FRAME_IN_ROBOT


def _compute_target_cam(data_dir, angle, scale, center):
    """Compute target camera pose in robot frame (same logic as simulation)."""
    poses = np.load(Path(data_dir) / "object_pose" / "poses.npz")["poses"]
    pos_cam_raw = poses[:, :3, 3]
    R = _cam_to_robot_matrix(angle)
    mean_raw = (R @ pos_cam_raw.T).T.mean(axis=0)
    cam_pos = (np.zeros(3) - mean_raw) * scale + np.array(center)

    # MuJoCo camera convention -> OpenCV convention
    # MuJoCo: X-right, Y-up, -Z-forward;  OpenCV: X-right, Y-down, Z-forward
    # mujoco_mat = [R[:,0], -R[:,1], -R[:,2]]
    # opencv_mat = mujoco_mat @ diag(1,-1,-1) = [R[:,0], R[:,1], R[:,2]] = R
    cam_rot = Rotation.from_matrix(R)
    return cam_pos, cam_rot


def align(env, data_dir, angle, scale, tcp_offset=0.097,
          pos_thresh=0.02, rot_thresh=5.0):
    """
    Guide user to align real 3rd-person camera to match simulation camera.
    Tracks camera in real-time and shows error relative to target pose.
    """
    if not BOARD_POSE_FILE.exists():
        print(f"Run 'locate_board' first. Missing: {BOARD_POSE_FILE}")
        return

    # get current robot TCP as center (same as simulation uses)
    obs = env.get_observation()
    state = obs["robot_state"]["cartesian_position"]
    flange_pos = np.array(state[:3])
    flange_rv = np.array(state[3:])
    R_flange = Rotation.from_rotvec(flange_rv).as_matrix()
    tcp_pos = flange_pos + tcp_offset * R_flange[:, 2]
    center = tuple(tcp_pos + np.array([-0.1, 0.0, 0.1]))

    target_pos, target_rot = _compute_target_cam(data_dir, angle, scale, center)
    print(f"Target camera pos:  [{target_pos[0]:+.4f}, {target_pos[1]:+.4f}, {target_pos[2]:+.4f}]")
    print(f"Target camera rot:  {target_rot.as_rotvec().round(3)}")
    target_euler = target_rot.as_euler('xyz', degrees=True)
    print(f"Target camera euler: [{target_euler[0]:+.1f}, {target_euler[1]:+.1f}, {target_euler[2]:+.1f}]°")
    print(f"Thresholds: pos < {pos_thresh*100:.0f}cm, rot < {rot_thresh:.0f}°")
    print(f"Move the 3rd-person camera by hand. Press Ctrl+C to stop.\n")

    # read dataset camera resolution for image saving
    cam_json_path = data_dir / "camera.json"
    if cam_json_path.exists():
        cam_json = json.load(open(cam_json_path))
        ds_w, ds_h = cam_json["width"], cam_json["height"]
    else:
        ds_w, ds_h = 640, 480

    board = _load_json(BOARD_POSE_FILE)
    T_board2base = _pose_to_4x4(np.array(board["R"]), np.array(board["t"]))

    from PIL import Image
    align_img_path = Path(__file__).parent / "align_view.png"
    _last_save = 0.0

    varied_cam_key = None
    intr_key = None

    try:
        while True:
            obs = env.get_observation()

            if varied_cam_key is None:
                image_dict = obs.get("image", {})
                for key in image_dict:
                    if "36776608" in key and "_left" in key:
                        varied_cam_key = key
                        break
                if varied_cam_key is None:
                    for key in image_dict:
                        if "36776608" in key:
                            varied_cam_key = key
                            break
                for k in obs.get("camera_intrinsics", {}):
                    if "36776608" in k:
                        intr_key = k
                if varied_cam_key is None or intr_key is None:
                    print("Cannot find varied camera.")
                    return

            image = obs["image"][varied_cam_key]
            now = time.time()
            if now - _last_save >= 1.0:
                pil_img = Image.fromarray(image).resize((ds_w, ds_h))
                pil_img.save(align_img_path)
                _last_save = now

            K, dist = _get_intrinsics(obs, intr_key)
            result = _detect_and_pose(image, K, dist)

            if result is None or result["rvec"] is None:
                print("Board not visible — move camera so board is in view")
                time.sleep(0.2)
                continue

            rvec_board, tvec_board = result["rvec"], result["tvec"]
            R_board2cam = Rotation.from_rotvec(rvec_board).as_matrix()
            T_board2cam = _pose_to_4x4(R_board2cam, tvec_board)
            T_cam2base = T_board2base @ np.linalg.inv(T_board2cam)

            cur_pos = T_cam2base[:3, 3]
            cur_rot = Rotation.from_matrix(T_cam2base[:3, :3])

            pos_err = np.linalg.norm(cur_pos - target_pos)
            rot_err = (target_rot.inv() * cur_rot).magnitude() * 180 / np.pi

            delta = target_pos - cur_pos
            dx, dy, dz = delta

            status = ""
            if pos_err < pos_thresh and rot_err < rot_thresh:
                status = " <<< ALIGNED! >>>"

            arrows = []
            if abs(dx) > pos_thresh / 2:
                arrows.append(f"{'→+X' if dx > 0 else '←-X'} {abs(dx)*100:.1f}cm")
            if abs(dy) > pos_thresh / 2:
                arrows.append(f"{'→+Y' if dy > 0 else '←-Y'} {abs(dy)*100:.1f}cm")
            if abs(dz) > pos_thresh / 2:
                arrows.append(f"{'↑+Z' if dz > 0 else '↓-Z'} {abs(dz)*100:.1f}cm")

            cur_euler = cur_rot.as_euler('xyz', degrees=True)
            print(f"pos_err: {pos_err*100:5.1f}cm  rot_err: {rot_err:5.1f}°  "
                  f"euler: [{cur_euler[0]:+.1f}, {cur_euler[1]:+.1f}, {cur_euler[2]:+.1f}]  "
                  f"move: {', '.join(arrows) if arrows else 'OK'}{status}")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopped.")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["calibrate_wrist", "locate_board", "track", "align"])
    parser.add_argument("--data-dir", default="data/035_power_drill_20200709_151335")
    parser.add_argument("--angle", type=float, default=45)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--tcp-offset", type=float, default=0.097)
    args = parser.parse_args()

    print("Creating RobotEnv...")
    env = _make_env()
    print("RobotEnv created.")
    if args.command == "calibrate_wrist":
        calibrate_wrist(env)
    elif args.command == "locate_board":
        locate_board(env)
    elif args.command == "track":
        track(env)
    elif args.command == "align":
        data_dir = Path(args.data_dir)
        if not data_dir.is_absolute():
            data_dir = PROJECT_ROOT / data_dir
        align(env, data_dir, args.angle, args.scale, tcp_offset=args.tcp_offset)
