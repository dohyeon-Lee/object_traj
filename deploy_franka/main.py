# initialize droid robot env

from droid.robot_env import RobotEnv
import time
import re
import json
import numpy as np
import imageio
import cv2
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _load_config():
    path = PROJECT_ROOT / "config.yml"
    if path.exists():
        return yaml.safe_load(path.read_text()) or {}
    return {}
DATASET_CAM_FRAME_IN_ROBOT = np.array([[0, 0, -1],
                                    [1, 0,  0],
                                    [0, -1, 0]], dtype=float)

_EEF_DIR_ROT = {
    'mz': Rotation.identity(),
    'py': Rotation.from_euler('y',  90, degrees=True) * Rotation.from_euler('x',  90, degrees=True),
    'my': Rotation.from_euler('y',  90, degrees=True) * Rotation.from_euler('x', -90, degrees=True),
}

def _parse_eef_dir(s):
    """Parse --eef-dir: 'mz'/'py'/'my' or 'SPH_lat<a>lon<b>[z<c>]'.

    SPH: lat=0 → +Z (top), lat=180 → -Z (=mz), +lon → CCW from +X toward +Y.
    Optional z suffix spins around the approach axis.
    """
    if s in _EEF_DIR_ROT:
        return _EEF_DIR_ROT[s]

    if s.upper().startswith('SPH_'):
        raw   = s[4:].lower()
        lat_m = re.search(r'lat(-?\d+(?:\.\d+)?)', raw)
        lon_m = re.search(r'lon(-?\d+(?:\.\d+)?)', raw)
        z_m   = re.search(r'z(-?\d+(?:\.\d+)?)',   raw)
        if not lat_m or not lon_m:
            raise ValueError(f"Invalid SPH format {s!r}. Use e.g. 'SPH_lat90lon0' or 'SPH_lat90lon0z10'.")
        lat   = float(lat_m.group(1))
        lon   = float(lon_m.group(1))
        z_rot = float(z_m.group(1)) if z_m else 0.0
        # approach dir d = [sin(lat)cos(lon), sin(lat)sin(lon), cos(lat)]; R @ [0,0,-1] = d
        return (Rotation.from_euler('z', lon, degrees=True)
                * Rotation.from_euler('y', lat + 180, degrees=True)
                * Rotation.from_euler('z', z_rot, degrees=True))

    raise ValueError(f"Invalid --eef-dir {s!r}. Use 'mz'/'py'/'my' or 'SPH_lat<a>lon<b>[z<c>]'.")

def cam_to_robot_matrix(angle_deg, R0=DATASET_CAM_FRAME_IN_ROBOT):
    return Rotation.from_euler('z', -angle_deg, degrees=True).as_matrix() @ R0

def cam_to_robot(pos, quat, R):
    return (R @ pos.T).T, (Rotation.from_matrix(R) * Rotation.from_quat(quat)).as_quat()

def tcp_to_flange(pos, euler_xyz, d):
    """Gripper-tip pose -> flange pose (subtract offset along flange Z)."""
    R = Rotation.from_euler("xyz", euler_xyz).as_matrix()
    return pos - d * R[:, 2], euler_xyz

def flange_to_tcp(pos, euler_xyz, d):
    """Flange pose -> gripper-tip pose (add offset along flange Z)."""
    R = Rotation.from_euler("xyz", euler_xyz).as_matrix()
    return pos + d * R[:, 2], euler_xyz

def remap(pos, quat, center=(0.0, 0.0, 1.0), scale=1.0):
    return (pos - pos.mean(axis=0)) * scale + np.array(center), quat

def load_traj(data_dir):
    poses = np.load(Path(data_dir) / "object_pose" / "poses.npz")["poses"]
    return poses[:, :3, 3], Rotation.from_matrix(poses[:, :3, :3]).as_quat()

# how to control the robot eef pose
def run(env, pose6, step=1, grip_close=False, hz=10):
    """
        pose6: [x,y,z,rx,ry,rz]
        grip_close: True==Close / False==Open
    """
    pose= np.array(pose6, dtype=np.float32)
    grip= np.array([1.0 if grip_close else 0.0], dtype=np.float32)
    action= np.concatenate([pose, grip], axis=0)
    
    for _ in range(step):
        env.step(action)
        time.sleep(1.0 / hz)


if __name__ == "__main__":
    cfg = _load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir",    nargs="?", default=cfg.get("data_dir", "data/035_power_drill_20200709_151335"))
    parser.add_argument("--angle",     type=float, default=cfg.get("angle", 90))
    parser.add_argument("--scale",     type=float, default=cfg.get("scale", 1))
    parser.add_argument("--eef-dir",   default=cfg.get("eef_dir", "mz"),
                        help="gripper approach: 'mz'/'py'/'my' or 'SPH_lat<a>lon<b>[z<c>]' (e.g. 'SPH_lat180lon0' for top-down)")
    parser.add_argument("--tcp-offset", type=float, default=cfg.get("tcp_offset", 0.145),
                        help="distance from flange to gripper tip along flange Z (meters)")
    args = parser.parse_args()
    
    action_space = "cartesian_position"
    gripper_action_space = "position"

    camera_kwargs = dict(
        hand_camera=dict(image=True, concatenate_images=False, resize_func="cv2"),
        varied_camera=dict(image=True, concatenate_images=False, resize_func="cv2"),
    )
    
    env = RobotEnv(
        action_space=action_space,
        gripper_action_space=gripper_action_space,
        camera_kwargs=camera_kwargs
    )

    env.reset(randomize=False)
    obs = env.get_observation()
    initial_pose = obs["robot_state"]["cartesian_position"]
    print("Joint angles after reset:", obs["robot_state"]["joint_positions"])
    print("Cartesian position (flange):", initial_pose[:3])

    tcp_d = args.tcp_offset
    tip_pos, _ = flange_to_tcp(initial_pose[:3], initial_pose[3:], tcp_d)
    print("Cartesian position (tcp):", tip_pos)

    # set the initial pose to the center of the trajectory
    center = tuple(tip_pos)
    center = tuple(np.array(center) + np.array(cfg.get("center_offset", [-0.1, 0.0, 0.1])))
    
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    R = cam_to_robot_matrix(args.angle)
    pos_cam, quat_cam = load_traj(data_dir)
    pos, quat = cam_to_robot(pos_cam, quat_cam, R)
    pos, quat = remap(pos, quat, center=center, scale=args.scale)

    rot_eef_init      = Rotation.from_euler("xyz", initial_pose[3:])

    rot_dataset_first = Rotation.from_quat(quat[0])
    rot_first = _parse_eef_dir(args.eef_dir) * rot_eef_init
    rot_first_vec = rot_first.as_euler("xyz")

    # --- video recording setup (varied camera, dataset resolution) ---
    cam_json_path = data_dir / "camera.json"
    if cam_json_path.exists():
        with open(cam_json_path) as f:
            cam_cfg = json.load(f)
        vid_w, vid_h = cam_cfg["width"], cam_cfg["height"]
    else:
        vid_w, vid_h = 640, 480

    varied_cam_key = None
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
    if varied_cam_key is None:
        print("WARNING: varied camera (36776608) not found, skipping video recording")
        print("  Available image keys:", list(image_dict.keys()))

    from PIL import Image
    data_name = Path(args.data_dir).name if Path(args.data_dir).name else data_dir.name
    video_dir = PROJECT_ROOT / "videos" / data_name
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"deploy_video_{int(args.angle)}_{args.eef_dir}.mp4"
    writer = imageio.get_writer(
        str(video_path), fps=10, codec="libx264", pixelformat="yuv420p",
    ) if varied_cam_key else None
    if writer:
        print(f"Recording video ({vid_w}x{vid_h}) → {video_path}")

    def _grab_frame():
        if writer is None:
            return
        frame_obs = env.get_observation()
        frame = frame_obs["image"][varied_cam_key]
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame).resize((vid_w, vid_h))
        writer.append_data(np.array(frame_pil))

    # --- move to first pose ---
    flange_pos0, _ = tcp_to_flange(pos[0], rot_first_vec, tcp_d)
    pose = np.concatenate([flange_pos0, rot_first_vec])
    run(env, pose.tolist(), step=100, grip_close=False)
    print("pose[0]:", pose)

    # --- Step 1: camera preview ---
    import threading
    preview_path = Path(__file__).parent / "camera_preview.png"
    if varied_cam_key:
        print(f"\n=== Camera Preview ===")
        print(f"Live view: {preview_path}")
        print(f"Press Enter to close gripper...\n")
        stop_preview = threading.Event()
        def _wait_enter():
            input()
            stop_preview.set()
        threading.Thread(target=_wait_enter, daemon=True).start()
        first_frame = True
        while not stop_preview.is_set():
            prev_obs = env.get_observation()
            frame = prev_obs["image"][varied_cam_key]
            if first_frame:
                print(f"  frame shape={frame.shape}, dtype={frame.dtype}, strides={frame.strides}, contiguous={frame.flags['C_CONTIGUOUS']}")
                first_frame = False
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(frame).resize((vid_w, vid_h)).save(preview_path)
            time.sleep(0.5)
    else:
        input("Press Enter to close gripper...")

    # --- Step 2: close gripper ---
    print("Closing gripper...")
    run(env, np.concatenate([flange_pos0, rot_first_vec]).tolist(), step=10, grip_close=True)
    print("Gripper closed. Press Enter to start trajectory...")
    input()

    # --- Step 3: trajectory loop ---
    _grab_frame()

    for i in range(len(pos)):
        rot_target = Rotation.from_quat(quat[i]) * rot_dataset_first.inv() * rot_first
        rot_target_vec = rot_target.as_euler("xyz")
        flange_pos_i, _ = tcp_to_flange(pos[i], rot_target_vec, tcp_d)
        pose = np.concatenate([flange_pos_i, rot_target_vec])
        print(pose)
        run(env, pose.tolist(), step=1, grip_close=True)
        _grab_frame()

    if writer:
        writer.close()
        print(f"Video saved → {video_path}")
        