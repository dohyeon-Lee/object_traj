# initialize droid robot env

from droid.robot_env import RobotEnv
import time
import numpy as np 
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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

def remap(pos, quat, center=(0.0, 0.0, 1.0), scale=1.0):
    return (pos - pos.mean(axis=0)) * scale + np.array(center), quat

def load_traj(data_dir):
    poses = np.load(Path(data_dir) / "object_pose" / "poses.npz")["poses"]
    return poses[:, :3, 3], Rotation.from_matrix(poses[:, :3, :3]).as_quat()

# how to control the robot eef pose
def run(env, pose6, duration=1.0, grip_close=False, hz=10):
    """
        pose6: [x,y,z,rx,ry,rz]
        grip_close: True==Close / False==Open
    """
    pose= np.array(pose6, dtype=np.float32)
    grip= np.array([1.0 if grip_close else 0.0], dtype=np.float32)
    action= np.concatenate([pose, grip], axis=0)

    for _ in range(int(duration * hz)):
        env.step(action)
        time.sleep(1.0 / hz)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir",    nargs="?", default="data/006_mustard_bottle_20200709_143211")
    parser.add_argument("--angle",     type=float, default=0.0)
    parser.add_argument("--scale",     type=float, default=1)
    parser.add_argument("--eef-dir",   default='mz',
                        help="gripper approach: 'mz'/'py'/'my' or 'SPH_lat<a>lon<b>[z<c>]' (e.g. 'SPH_lat180lon0' for top-down)")
    args = parser.parse_args()
    
    action_space = "cartesian_position"
    gripper_action_space = "position"

    env = RobotEnv(
        action_space=action_space,
        gripper_action_space=gripper_action_space,
        # camera_kwargs={}
    )

    env.reset(randomize=False)
    obs = env.get_observation()
    initial_pose = obs["robot_state"]["cartesian_position"]
    
    center = tuple(initial_pose[:3])
    
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    R = cam_to_robot_matrix(args.angle)
    pos_cam, quat_cam = load_traj(data_dir)
    pos, quat = cam_to_robot(pos_cam, quat_cam, R)
    pos, quat = remap(pos, quat, center=center, scale=args.scale)
    
    rot_eef_init      = Rotation.from_rotvec(initial_pose[3:])
    rot_dataset_first = Rotation.from_quat(quat[0])
    rot_first = _parse_eef_dir(args.eef_dir) * rot_eef_init
    rot_first_vec =rot_first.as_rotvec()
    
    pose = np.concatenate([np.array(pos[0]), np.array(rot_first_vec)])
    
    # run(env, pose.tolist(), duration=1.0, grip_close=False)
    
    # for i in range(len(pos)):
    #     rot_target = Rotation.from_quat(quat[i]) * rot_dataset_first.inv() * rot_first
    #     rot_target_vec = rot_target.as_rotvec()
    #     pose = np.concatenate([np.array(pos[i]), np.array(rot_target_vec)])
    #     print(pose)
    #     run(env, pose.tolist(), duration=0.1, grip_close=False)
        
        
    # run(env, pose.tolist(), duration=1.0, grip_close=False)
    # time.sleep(1.0)
    # run(env, initial_pos.tolist(), duration=2.0, grip_close=False)
    # time.sleep(1.0)
    # run(env, y.tolist(), duration=2.0, grip_close=False)
    # time.sleep(1.0)
    # run(env, initial_pos.tolist(), duration=2.0, grip_close=False)
    # time.sleep(1.0)
    # run(env, z.tolist(), duration=2.0, grip_close=False)
    # time.sleep(1.0)
    # run(env, initial_pos.tolist(), duration=2.0, grip_close=False)