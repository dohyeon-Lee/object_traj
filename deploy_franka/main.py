# initialize droid robot env

from droid.robot_env import RobotEnv
import time
import re
import numpy as np
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
        return (Rotation.from_euler('z', lon, degrees=True)
                * Rotation.from_euler('y', lat + 180, degrees=True)
                * Rotation.from_euler('z', z_rot, degrees=True))

    raise ValueError(f"Invalid --eef-dir {s!r}. Use 'mz'/'py'/'my' or 'SPH_lat<a>lon<b>[z<c>]'.")

def cam_to_robot_matrix(angle_deg, elevation_deg=0.0, R0=DATASET_CAM_FRAME_IN_ROBOT):
    elev = Rotation.from_euler('x', -elevation_deg, degrees=True).as_matrix()
    return Rotation.from_euler('z', -angle_deg, degrees=True).as_matrix() @ R0 @ elev

def cam_to_robot(pos, quat, R):
    return (R @ pos.T).T, (Rotation.from_matrix(R) * Rotation.from_quat(quat)).as_quat()

def tcp_to_flange(pos, euler_xyz, d):
    R = Rotation.from_euler("xyz", euler_xyz).as_matrix()
    return pos - d * R[:, 2], euler_xyz

def flange_to_tcp(pos, euler_xyz, d):
    R = Rotation.from_euler("xyz", euler_xyz).as_matrix()
    return pos + d * R[:, 2], euler_xyz

def remap(pos, quat, center=(0.0, 0.0, 1.0), scale=1.0, use_initial=False):
    ref = pos[0] if use_initial else pos.mean(axis=0)
    return (pos - ref) * scale + np.array(center), quat

def load_traj(data_dir):
    poses = np.load(Path(data_dir) / "object_pose" / "poses.npz")["poses"]
    return poses[:, :3, 3], Rotation.from_matrix(poses[:, :3, :3]).as_quat()

def run(env, pose6, step=1, grip_close=False, hz=10):
    pose= np.array(pose6, dtype=np.float32)
    grip= np.array([1.0 if grip_close else 0.0], dtype=np.float32)
    action= np.concatenate([pose, grip], axis=0)

    for _ in range(step):
        env.step(action)
        time.sleep(1.0 / hz)


if __name__ == "__main__":
    cfg = _load_config()
    initial_frame = cfg.get("start_frame", 0)
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir",    nargs="?", default=cfg.get("data_dir", "data/035_power_drill_20200709_151335"))
    parser.add_argument("--angle",     type=float, default=cfg.get("angle", 90))
    parser.add_argument("--elevation", type=float, default=cfg.get("elevation", 0.0),
                        help="vertical camera angle in degrees (0=horizontal, 90=top-down)")
    parser.add_argument("--scale",     type=float, default=cfg.get("scale", 1))
    parser.add_argument("--eef-dir",   default=cfg.get("eef_dir", "mz"),
                        help="gripper approach: 'mz'/'py'/'my' or 'SPH_lat<a>lon<b>[z<c>]' (e.g. 'SPH_lat180lon0' for top-down)")
    parser.add_argument("--tcp-offset", type=float, default=cfg.get("tcp_offset", 0.145),
                        help="distance from flange to gripper tip along flange Z (meters)")
    parser.add_argument("--initial",    action="store_true", default=cfg.get("initial", False),
                        help="anchor traj start (frame 0) to center instead of mean")
    args = parser.parse_args()

    action_space = "cartesian_position"
    gripper_action_space = "position"

    env = RobotEnv(
        action_space=action_space,
        gripper_action_space=gripper_action_space,
    )

    env.reset(randomize=False)
    obs = env.get_observation()
    initial_pose = obs["robot_state"]["cartesian_position"]
    print("Joint angles after reset:", obs["robot_state"]["joint_positions"])
    print("Cartesian position (flange):", initial_pose[:3])

    tcp_d = args.tcp_offset
    tip_pos, _ = flange_to_tcp(initial_pose[:3], initial_pose[3:], tcp_d)
    print("Cartesian position (tcp):", tip_pos)

    center = tuple(tip_pos)
    center = tuple(np.array(center) + np.array(cfg.get("center_offset", [-0.1, 0.0, 0.1])))

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    R = cam_to_robot_matrix(args.angle, args.elevation)
    pos_cam, quat_cam = load_traj(data_dir)
    pos, quat = cam_to_robot(pos_cam, quat_cam, R)
    pos, quat = remap(pos, quat, center=center, scale=args.scale, use_initial=args.initial)

    rot_eef_init      = Rotation.from_euler("xyz", initial_pose[3:])

    rot_dataset_first = Rotation.from_quat(quat[initial_frame])
    rot_first = _parse_eef_dir(args.eef_dir) * rot_eef_init
    rot_first_vec = rot_first.as_euler("xyz")

    # --- move to first pose ---
    flange_pos0, _ = tcp_to_flange(pos[initial_frame], rot_first_vec, tcp_d)
    pose = np.concatenate([flange_pos0, rot_first_vec])
    run(env, pose.tolist(), step=100, grip_close=False)
    print("pose[0]:", pose)

    # --- close gripper ---
    input("Press Enter to close gripper...")
    print("Closing gripper...")
    run(env, np.concatenate([flange_pos0, rot_first_vec]).tolist(), step=10, grip_close=True)
    print("Gripper closed. Press Enter to start trajectory...")
    input()

    # --- trajectory loop ---
    for i in range(len(pos) - initial_frame):
        rot_target = Rotation.from_quat(quat[i + initial_frame]) * rot_dataset_first.inv() * rot_first
        rot_target_vec = rot_target.as_euler("xyz")
        flange_pos_i, _ = tcp_to_flange(pos[i + initial_frame], rot_target_vec, tcp_d)
        pose = np.concatenate([flange_pos_i, rot_target_vec])
        print(pose)
        run(env, pose.tolist(), step=1, grip_close=True)
