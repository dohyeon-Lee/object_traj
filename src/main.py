# Usage (run from project root, activate object_traj venv first):
#   source .venv/bin/activate
#   python src/main.py data/035_power_drill_20200709_151335 --no-wandb --show-eef --angle 45 --eef-dir my
#
# Robot starts directly at the first trajectory pose via IK (no warm-up phase).

import os
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

# Project root = one level up from src/main.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("MUJOCO_GL", "egl")

import json
import math
import imageio
import numpy as np
import wandb
import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config
from scipy.spatial.transform import Rotation

from viz_overlay import get_cam_matrices, draw_eef, make_video

CAMERAS     = ["frontview", "birdview", "sideview"]
DATASET_CAM = "dataset_cam"

# Camera-to-robot rotation matrix (empirically verified).
# Assumption: dataset camera faces the robot head-on in simulation (+X direction).
# cam +Z (optical axis) = robot -X
# cam +Y (down)         = robot -Z
# cam +X (right)        = robot +Y
DATASET_CAM_FRAME_IN_ROBOT = np.array( [[ 0,  0, -1],
                                        [ 1,  0,  0],
                                        [ 0, -1,  0]], dtype=float)

KP_POS = 5.0   # proportional gain for position error (m → action)
KP_ROT = 2.0   # proportional gain for rotation error (rad → action)

# World-frame rotation applied to EEF init to orient the gripper at trajectory start.
# Each entry rotates the default approach direction (-Z_robot) to the target direction.
_EEF_DIR_ROT = {
    'mz': Rotation.identity(),
    'py': Rotation.from_euler('y',  90, degrees=True) * Rotation.from_euler('x',  90, degrees=True),
    'my': Rotation.from_euler('y',  90, degrees=True) * Rotation.from_euler('x', -90, degrees=True),
}


def cam_to_robot_matrix(angle_deg: float, R0=DATASET_CAM_FRAME_IN_ROBOT) -> np.ndarray:
    """Cam-to-robot rotation matrix for a given horizontal camera angle.

    Angle is defined from the robot's perspective:
    0°   = camera directly in front (head-on)
    +90° = camera to the robot's right
    -90° = camera to the robot's left
    """
    rot_z = Rotation.from_euler('z', -angle_deg, degrees=True).as_matrix()
    return rot_z @ R0


# ── data ──────────────────────────────────────────────────────────────────────

def load_traj(data_dir):
    """poses.npz → (N,3) pos, (N,4) quat  in camera frame"""
    poses = np.load(Path(data_dir) / "object_pose" / "poses.npz")["poses"]
    pos  = poses[:, :3, 3]
    quat = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    return pos, quat


def cam_to_robot(pos, quat, R=DATASET_CAM_FRAME_IN_ROBOT):
    """camera frame → robot frame using R (default: DATASET_CAM_FRAME_IN_ROBOT)"""
    rot = Rotation.from_matrix(R)
    return (R @ pos.T).T, (rot * Rotation.from_quat(quat)).as_quat()


def remap(pos, quat, center=(0.0, 0.0, 1.0), scale=1.0):
    """shift + scale so trajectory center lands at `center` in robosuite world"""
    pos = (pos - pos.mean(axis=0)) * scale + np.array(center)
    return pos, quat


# ── dataset camera pose ───────────────────────────────────────────────────────

def compute_dataset_cam(pos_cam_raw, scale, center=(0.0, 0.0, 1.0), R=DATASET_CAM_FRAME_IN_ROBOT):
    """Compute dataset camera position and MuJoCo quaternion in robot frame.

    The camera origin (0,0,0 in cam frame) undergoes the same remap as the
    trajectory points, telling us where the real camera sits in robot world.
    """
    mean_raw = (R @ pos_cam_raw.T).T.mean(axis=0)
    cam_pos  = (np.zeros(3) - mean_raw) * scale + np.array(center)

    # MuJoCo cam frame from R (columns = cam axes in robot frame):
    #   X_cam=R[:,0], Y_cam(up)=-R[:,1], Z_cam(backward)=-R[:,2]
    mujoco_cam_mat = np.column_stack([R[:, 0], -R[:, 1], -R[:, 2]])
    qxyzw = Rotation.from_matrix(mujoco_cam_mat).as_quat()
    cam_quat_wxyz = np.array([qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]])
    return cam_pos, cam_quat_wxyz


# ── mesh utils ────────────────────────────────────────────────────────────────

def _clean_obj(src: Path, dst: Path):
    """Strip vertex colors from OBJ (v x y z r g b → v x y z). Keeps UV coords."""
    with open(src) as fin, open(dst, 'w') as fout:
        for line in fin:
            if line.startswith('v '):
                parts = line.split()
                fout.write(f"v {parts[1]} {parts[2]} {parts[3]}\n")
            else:
                fout.write(line)


# ── IK ────────────────────────────────────────────────────────────────────────

def solve_ik(env, target_pos, target_rot, n_iter=1000, tol=1e-4, damping=0.01, max_dq=0.2):
    """Jacobian pseudo-inverse IK targeting the EEF body (consistent with obs eef_quat)."""
    import mujoco
    robot = env.robots[0]
    m = getattr(env.sim.model, '_model', env.sim.model)
    d = getattr(env.sim.data,  '_data',  env.sim.data)

    # grip_site: center between fingers = obs["robot0_eef_pos"] source
    site_id_raw = robot.eef_site_id
    site_id = next(iter(site_id_raw.values())) if isinstance(site_id_raw, dict) else int(site_id_raw)

    # eef body: orientation source = obs["robot0_eef_quat"] source
    eef_name_raw  = robot.robot_model.eef_name
    eef_body_name = next(iter(eef_name_raw.values())) if isinstance(eef_name_raw, dict) else eef_name_raw
    body_id   = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, eef_body_name)

    joint_ids = list(robot._ref_joint_pos_indexes)
    nv        = m.nv
    jacp      = np.zeros((3, nv))
    jacr      = np.zeros((3, nv))
    _dummy    = np.zeros((3, nv))
    R_target  = target_rot.as_matrix()
    err       = np.ones(6)

    for _ in range(n_iter):
        mujoco.mj_forward(m, d)
        dp  = target_pos - d.site_xpos[site_id]
        dR  = Rotation.from_matrix(R_target @ d.xmat[body_id].reshape(3, 3).T).as_rotvec()
        err = np.concatenate([dp, dR])
        if np.linalg.norm(err) < tol:
            break
        mujoco.mj_jacSite(m, d, jacp,   _dummy, site_id)  # position rows from grip_site
        mujoco.mj_jacBody(m, d, _dummy, jacr,   body_id)  # rotation rows from eef body
        J  = np.vstack([jacp, jacr])[:, joint_ids]
        dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), err)
        d.qpos[joint_ids] += np.clip(dq, -max_dq, max_dq)

    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    print(f"IK residual: {np.linalg.norm(err):.5f}")


# ── simulation ────────────────────────────────────────────────────────────────

def run_sim(pos, quat, pos_cam_raw, scale, center,
            steps_per_waypoint, video_dir, run_name, wandb_run, R=DATASET_CAM_FRAME_IN_ROBOT,
            angle=0.0, eef_dir='mz', show_eef=False, fovy=60.0, cam_h=480, cam_w=640,
            data_dir=None):

    cam_pos, cam_quat = compute_dataset_cam(pos_cam_raw, scale, center, R=R)
    dataset_cam_key = f"dataset_cam_{angle:g}_{eef_dir}"

    # hard_reset=False: reset() calls sim.reset() only, not _load_model()/_initialize_sim()
    # This lets us inject a custom camera via _initialize_sim(new_xml) without it being overwritten.
    env = suite.make(
        env_name="Lift", robots="Panda",
        controller_configs=load_composite_controller_config(robot="panda"),
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
        camera_names=CAMERAS, camera_heights=cam_h, camera_widths=cam_w,
        ignore_done=True, horizon=steps_per_waypoint * len(pos) + 100,
        hard_reset=False,
    )

    # Inject dataset camera + object mesh: parse current XML, add elements, recompile
    xml  = env.model.get_xml()
    root = ET.fromstring(xml)

    ET.SubElement(root.find('worldbody'), 'camera', {
        'name': DATASET_CAM,
        'pos':  ' '.join(f"{v:.4f}" for v in cam_pos),
        'quat': ' '.join(f"{v:.6f}" for v in cam_quat),
        'fovy': f'{fovy:.4f}',
    })

    if data_dir is not None:
        mesh_dir  = Path(data_dir) / "mesh"
        clean_obj = mesh_dir / "textured_simple_clean.obj"
        if not clean_obj.exists():
            _clean_obj(mesh_dir / "textured_simple.obj", clean_obj)
        tex_file  = str(mesh_dir / "texture_map.png")
        asset = root.find('asset')
        if asset is None:
            asset = ET.SubElement(root, 'asset')
        ET.SubElement(asset, 'mesh',     {'name': 'obj_mesh', 'file': str(clean_obj)})
        ET.SubElement(asset, 'texture',  {'name': 'obj_tex', 'type': '2d', 'file': tex_file})
        ET.SubElement(asset, 'material', {'name': 'obj_mat', 'texture': 'obj_tex',
                                          'texuniform': 'false', 'specular': '0.3'})
        obj_quat_wxyz = [quat[0][3], quat[0][0], quat[0][1], quat[0][2]]
        obj_body = ET.SubElement(root.find('worldbody'), 'body', {
            'name': 'traj_object',
            'pos': ' '.join(f'{v:.4f}' for v in pos[0]),
            'quat': ' '.join(f'{v:.6f}' for v in obj_quat_wxyz),
        })
        ET.SubElement(obj_body, 'freejoint', {'name': 'traj_object_joint'})
        ET.SubElement(obj_body, 'inertial', {
            'pos': '0 0 0',
            'mass': '0.1',
            'diaginertia': '0.001 0.001 0.001',
        })
        ET.SubElement(obj_body, 'geom', {
            'type': 'mesh', 'mesh': 'obj_mesh', 'material': 'obj_mat',
            'group': '1', 'contype': '1', 'conaffinity': '1', 'friction': '2 0.05 0.01',
        })

    env._initialize_sim(ET.tostring(root, encoding='unicode'))

    obs = env.reset()  # hard_reset=False → sim.reset() only, keeps our compiled model
    print(f"Dataset cam pos (robot frame): {cam_pos.round(3)}")

    rot_eef_init      = Rotation.from_quat(obs["robot0_eef_quat"].copy())
    rot_dataset_first = Rotation.from_quat(quat[0])
    rot_first = _EEF_DIR_ROT[eef_dir] * rot_eef_init
    C_inv     = rot_first.inv() * rot_dataset_first

    solve_ik(env, pos[0], rot_first)
    obs = env._get_observations()

    # free-joint object handles (-1 if object not injected)
    obj_body_id = -1
    obj_qpos_adr = -1
    obj_dof_adr = -1
    if data_dir is not None:
        import mujoco
        m = getattr(env.sim.model, '_model', env.sim.model)
        d = getattr(env.sim.data,  '_data',  env.sim.data)
        obj_body_id  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'traj_object')
        obj_joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'traj_object_joint')
        obj_qpos_adr = int(m.jnt_qposadr[obj_joint_id])
        obj_dof_adr  = int(m.jnt_dofadr[obj_joint_id])
        print(f"Object body_id={obj_body_id}  qpos_adr={obj_qpos_adr}  dof_adr={obj_dof_adr}  pos[0]={pos[0].round(3)}")

    def _set_object_pose(p, q_xyzw):
        import mujoco
        if obj_qpos_adr < 0:
            return
        m = getattr(env.sim.model, '_model', env.sim.model)
        d = getattr(env.sim.data,  '_data',  env.sim.data)
        d.qpos[obj_qpos_adr:obj_qpos_adr + 3] = p
        d.qpos[obj_qpos_adr + 3:obj_qpos_adr + 7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
        d.qvel[obj_dof_adr:obj_dof_adr + 6] = 0
        mujoco.mj_forward(m, d)
    if obj_qpos_adr >= 0:
        _set_object_pose(pos[0], quat[0])

    # Phase 1: keep object fixed at first pose while gripper closes around it.
    for _ in range(100):
        if obj_qpos_adr >= 0:
            _set_object_pose(pos[0], quat[0])
        dp = np.clip((pos[0] - obs["robot0_eef_pos"]) * KP_POS, -1, 1)
        dR = np.clip((rot_first * Rotation.from_quat(obs["robot0_eef_quat"]).inv()).as_rotvec() * KP_ROT, -1, 1)
        obs, *_ = env.step(np.concatenate([dp, dR, [1.0]]))

    rel_pos_in_eef = None
    rel_rot = None
    if obj_qpos_adr >= 0:
        import mujoco
        m = getattr(env.sim.model, '_model', env.sim.model)
        d = getattr(env.sim.data,  '_data',  env.sim.data)
        eef_name_raw  = env.robots[0].robot_model.eef_name
        eef_body_name = next(iter(eef_name_raw.values())) if isinstance(eef_name_raw, dict) else eef_name_raw
        eef_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, eef_body_name)

        eef_pos = d.xpos[eef_body_id].copy()
        eef_rot = Rotation.from_matrix(d.xmat[eef_body_id].reshape(3, 3))
        obj_pos = d.xpos[obj_body_id].copy()
        obj_quat_wxyz = d.xquat[obj_body_id].copy()
        obj_rot = Rotation.from_quat([obj_quat_wxyz[1], obj_quat_wxyz[2], obj_quat_wxyz[3], obj_quat_wxyz[0]])
        rel_pos_in_eef = eef_rot.inv().apply(obj_pos - eef_pos)
        rel_rot = eef_rot.inv() * obj_rot

    cam_mat     = {cam: get_cam_matrices(env, cam, cam_h, cam_w) for cam in CAMERAS + [DATASET_CAM]}
    frames      = {cam: [] for cam in CAMERAS + [dataset_cam_key]}
    eef_history = []

    for i, (tgt_pos, tgt_quat) in enumerate(zip(pos, quat)):
        for _ in range(steps_per_waypoint):
            delta_pos  = np.clip((tgt_pos - obs["robot0_eef_pos"]) * KP_POS, -1, 1)
            rot_target = Rotation.from_quat(tgt_quat) * rot_dataset_first.inv() * rot_first
            r_delta    = rot_target * Rotation.from_quat(obs["robot0_eef_quat"]).inv()
            delta_rot  = np.clip(r_delta.as_rotvec() * KP_ROT, -1, 1)
            action     = np.concatenate([delta_pos, delta_rot, [1.0]])

            obs, *_ = env.step(action)

            # Phase 2: after grasp, carry object with EEF using fixed relative transform.
            if obj_qpos_adr >= 0:
                eef_now = Rotation.from_quat(obs["robot0_eef_quat"])
                obj_pos_now = obs["robot0_eef_pos"] + eef_now.apply(rel_pos_in_eef)
                obj_quat_now = (eef_now * rel_rot).as_quat()
                _set_object_pose(obj_pos_now, obj_quat_now)

            eef_history.append(obs["robot0_eef_pos"].copy())

            eef_quat_vis = (Rotation.from_quat(obs["robot0_eef_quat"]) * C_inv).as_quat()
            for cam in CAMERAS:
                img = obs[f"{cam}_image"][::-1].copy()
                K, cp, cr = cam_mat[cam]
                if show_eef:
                    draw_eef(img, eef_history, eef_quat_vis, K, cp, cr)
                frames[cam].append(img)

            # Render dataset camera manually (not in robosuite obs pipeline)
            ds_img = env.sim.render(cam_w, cam_h, camera_name=DATASET_CAM)[::-1].copy()
            if show_eef:
                K, cp, cr = cam_mat[DATASET_CAM]
                draw_eef(ds_img, eef_history, eef_quat_vis, K, cp, cr)
            frames[dataset_cam_key].append(ds_img)

        print(f"[{i+1}/{len(pos)}] err={np.linalg.norm(obs['robot0_eef_pos'] - tgt_pos):.4f}")

    env.close()
    _save(frames, video_dir, run_name, wandb_run)


def _save(frames, video_dir, run_name, wandb_run):
    out_dir = Path(video_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for cam, cam_frames in frames.items():
        path = str(out_dir / f"{cam}.mp4")
        imageio.mimwrite(path, cam_frames, fps=20, codec="libx264",
                         output_params=["-crf", "18"])
        print(f"Saved {cam} -> {path}")
        if wandb_run:
            wandb_run.log({f"video/{cam}": wandb.Video(path, fps=20, format="mp4")})


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir",    nargs="?", default="data/006_mustard_bottle_20200709_143211")
    parser.add_argument("--scale",      type=float, default=1)
    parser.add_argument("--steps",      type=int,   default=2)
    parser.add_argument("--video-dir", default="videos")
    parser.add_argument("--project",   default="robosuite-eef-traj")
    parser.add_argument("--name",      default=None)
    parser.add_argument("--no-wandb",  action="store_true")
    parser.add_argument("--show-eef",  action="store_true", help="overlay EEF trail and orientation axes on video")
    parser.add_argument("--angle",     type=float, default=0.0, help="horizontal camera angle in degrees (0=head-on, +90=right, -90=left)")
    parser.add_argument("--eef-dir",   default='mz', choices=['mz', 'py', 'my'],
                        help="gripper approach direction at trajectory start in robot frame")
    args = parser.parse_args()

    center    = (0.0, 0.0, 1.0)
    data_dir  = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    run_name  = args.name or data_dir.name
    R = cam_to_robot_matrix(args.angle)
    pos_cam, quat_cam = load_traj(data_dir)
    pos, quat = cam_to_robot(pos_cam, quat_cam, R=R)
    pos, quat = remap(pos, quat, center=center, scale=args.scale)

    video_dir = Path(args.video_dir)
    if not video_dir.is_absolute():
        video_dir = PROJECT_ROOT / video_dir

    cam_json = json.load(open(data_dir / "camera.json"))
    fy    = cam_json["intrinsics"][1][1]
    cam_h = cam_json["height"]
    cam_w = cam_json["width"]
    fovy  = math.degrees(2 * math.atan(cam_h / (2 * fy)))

    make_video(data_dir)

    wandb_run = None if args.no_wandb else wandb.init(project=args.project, name=run_name)
    run_sim(pos, quat, pos_cam, args.scale, center,
            args.steps, video_dir, run_name, wandb_run, R=R,
            angle=args.angle, eef_dir=args.eef_dir, show_eef=args.show_eef,
            fovy=fovy, cam_h=cam_h, cam_w=cam_w, data_dir=data_dir)
    if wandb_run:
        wandb_run.finish()
