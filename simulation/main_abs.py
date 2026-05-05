# Usage (run from project root, activate object_traj venv first):
#   source .venv/bin/activate
#   python src/main.py data/freepose --no-wandb --show-eef --angle 45 --eef-dir my
#   python simulation/main.py data/035_power_drill_20200709_151335 --no-wandb --show-eef --angle 45 --eef-dir mz

import os
import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _load_config():
    path = PROJECT_ROOT / "config.yml"
    if path.exists():
        return yaml.safe_load(path.read_text()) or {}
    return {}
os.environ.setdefault("MUJOCO_GL", "egl")

import re
import imageio
import numpy as np
import wandb
import robosuite as suite
from robosuite.models.robots import Panda as PandaModel
from robosuite.models.robots.robot_model import register_robot
from robosuite.robots import ROBOT_CLASS_MAPPING
from scipy.spatial.transform import Rotation

# Panda model + Robotiq85 with 45° z-axis rotation at gripper mount
class PandaRotatedGripper(PandaModel):
    @property
    def gripper_mount_quat_offset(self):
        q = Rotation.from_euler('z', 45, degrees=True).as_quat()  # scipy [x,y,z,w]
        return {'right': np.array([q[3], q[0], q[1], q[2]])}      # MuJoCo [w,x,y,z]

register_robot(PandaRotatedGripper)
ROBOT_CLASS_MAPPING['PandaRotatedGripper'] = ROBOT_CLASS_MAPPING['Panda']

try:
    from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config
    def _load_ctrl_cfg():
        cfg = load_composite_controller_config(robot="panda")
        for part_cfg in cfg.get("body_parts", {}).values():
            if isinstance(part_cfg, dict) and part_cfg.get("type", "").upper().startswith("OSC"):
                part_cfg["input_type"] = "absolute"
                part_cfg["input_ref_frame"] = "world"
        return cfg
except Exception:
    from robosuite.controllers import load_controller_config
    def _load_ctrl_cfg():
        cfg = load_controller_config(default_controller="OSC_POSE")
        cfg["input_type"] = "absolute"
        cfg["input_ref_frame"] = "world"
        return cfg

from viz_overlay import (get_cam_matrices, draw_eef,
                        inject_object_xml, make_object_updater, make_video)

# ── constants ─────────────────────────────────────────────────────────────────

CAMERAS     = ["frontview", "birdview", "sideview_opp"]
DATASET_CAM = "dataset_cam"
SIDEVIEW_OPP = {
    "name": "sideview_opp",
    "pos":  "-0.056518 -1.276122 1.487957",
    "quat": "-0.806418 -0.591223 0.006878 0.009905",
}

# cam +Z (optical) = robot -X,  cam +Y (down) = robot -Z,  cam +X (right) = robot +Y
DATASET_CAM_FRAME_IN_ROBOT = np.array([[0, 0, -1],
                                        [1, 0,  0],
                                        [0, -1, 0]], dtype=float)


DROID_INIT_QPOS = np.array([0, -np.pi / 5, 0, -4 * np.pi / 5, 0, 3 * np.pi / 5, 0.0])

def _set_droid_init_qpos(env):
    import mujoco
    robot = env.robots[0]
    env.sim.data.qpos[robot._ref_joint_pos_indexes] = DROID_INIT_QPOS
    env.sim.data.qvel[robot._ref_joint_vel_indexes] = 0
    m = getattr(env.sim.model, '_model', env.sim.model)
    d = getattr(env.sim.data, '_data', env.sim.data)
    mujoco.mj_forward(m, d)

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


# ── data loading + transforms ─────────────────────────────────────────────────

def load_traj(data_dir):
    poses = np.load(Path(data_dir) / "object_pose" / "poses.npz")["poses"]
    return poses[:, :3, 3], Rotation.from_matrix(poses[:, :3, :3]).as_quat()


def cam_to_robot_matrix(angle_deg, elevation_deg=0.0, R0=DATASET_CAM_FRAME_IN_ROBOT):
    elev = Rotation.from_euler('x', -elevation_deg, degrees=True).as_matrix()
    return Rotation.from_euler('z', -angle_deg, degrees=True).as_matrix() @ R0 @ elev


def cam_to_robot(pos, quat, R):
    return (R @ pos.T).T, (Rotation.from_matrix(R) * Rotation.from_quat(quat)).as_quat()


def remap(pos, quat, center=(0.0, 0.0, 1.0), scale=1.0, use_initial=False):
    ref = pos[0] if use_initial else pos.mean(axis=0)
    return (pos - ref) * scale + np.array(center), quat


def compute_dataset_cam(pos_cam_raw, scale, center, R, use_initial=False, cam_distance=1.0):
    anchor   = pos_cam_raw[0] if use_initial else pos_cam_raw.mean(axis=0)
    mean_raw = R @ anchor
    cam_pos  = (np.zeros(3) - mean_raw) * scale + np.array(center)
    if cam_distance != 1.0:
        direction = cam_pos - np.array(center)
        cam_pos = np.array(center) + direction * cam_distance
    mujoco_cam_mat = np.column_stack([R[:, 0], -R[:, 1], -R[:, 2]])
    q = Rotation.from_matrix(mujoco_cam_mat).as_quat()
    return cam_pos, np.array([q[3], q[0], q[1], q[2]])


def _hide_default_lift_props(root):
    """Hide Lift table and cube without deleting them (robosuite references their names)."""
    hide_prefixes = ("table_", "cube_g0")
    for geom in root.findall(".//geom"):
        name = geom.get("name", "")
        if name.startswith(hide_prefixes):
            rgba = geom.get("rgba", "1 1 1 1").split()
            if len(rgba) != 4:
                rgba = ["1", "1", "1", "1"]
            rgba[3] = "0"
            geom.set("rgba", " ".join(rgba))
            geom.set("contype", "0")
            geom.set("conaffinity", "0")


def _set_white_background(root):
    """Set simulation background to white using a flat white skybox texture."""
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    for tex in asset.findall("texture"):
        if tex.get("type") == "skybox":
            tex.set("builtin", "flat")
            tex.set("rgb1", "1 1 1")
            tex.set("rgb2", "1 1 1")
            return
    ET.SubElement(asset, "texture", {
        "name": "white_skybox",
        "type": "skybox",
        "builtin": "flat",
        "rgb1": "1 1 1",
        "rgb2": "1 1 1",
        "width": "512",
        "height": "512",
    })


def _hide_robot_mount_and_floor(root):
    """Hide robot mount/stand/wall geoms and floor/ceiling planes."""
    hide_keywords = ["mount", "pedestal", "stand", "wall", "floor", "ceiling"]
    for geom in root.findall(".//geom"):
        name = geom.get("name", "").lower()
        geom_type = geom.get("type", "")
        if any(kw in name for kw in hide_keywords) or geom_type == "plane":
            rgba = geom.get("rgba", "1 1 1 1").split()
            if len(rgba) != 4:
                rgba = ["1", "1", "1", "1"]
            rgba[3] = "0"
            geom.set("rgba", " ".join(rgba))
            geom.set("contype", "0")
            geom.set("conaffinity", "0")


def _set_offscreen_framebuffer(root, width, height):
    """Ensure MuJoCo's offscreen framebuffer can render dataset-resolution videos."""
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_cfg = visual.find("global")
    if global_cfg is None:
        global_cfg = ET.SubElement(visual, "global")
    global_cfg.set("offwidth", str(int(width)))
    global_cfg.set("offheight", str(int(height)))


def move_to_start_pose(env, target_pos, target_rot, n_steps=100):
    """Move to the first pose through the controller, matching deploy_franka startup."""
    action = np.concatenate([target_pos, target_rot.as_rotvec(), [1.0]])
    obs = env._get_observations()
    for _ in range(n_steps):
        obs, *_ = env.step(action)
    return obs


def make_renderers(env, width, height, camera_names):
    """Create MuJoCo renderers with visual geoms only."""
    import mujoco

    model = getattr(env.sim.model, "_model", env.sim.model)
    data = getattr(env.sim.data, "_data", env.sim.data)
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[:] = 0
    scene_option.geomgroup[1] = 1
    camera_ids = {
        camera_name: env.sim.model.camera_name2id(camera_name)
        for camera_name in camera_names
    }
    renderers = {
        camera_name: mujoco.Renderer(model, height=height, width=width)
        for camera_name in camera_names
    }
    return renderers, data, scene_option, camera_ids


def render_camera(renderer, data, scene_option, camera_id, warmup=False):
    """Render one camera frame; warmup discards a stale offscreen buffer."""
    if warmup:
        renderer.update_scene(data, camera=camera_id, scene_option=scene_option)
        renderer.render()
    renderer.update_scene(data, camera=camera_id, scene_option=scene_option)
    return renderer.render().copy()


# ── simulation ────────────────────────────────────────────────────────────────

def run_sim(pos, quat, pos_cam_raw, scale, center,
            steps_per_waypoint, video_dir, run_name, wandb_run,
            R=DATASET_CAM_FRAME_IN_ROBOT, angle=0.0, elevation=0.0, eef_dir='mz',
            show_eef=False, fovy=60.0, cam_h=480, cam_w=640, data_dir=None,
            mesh_rot_offset=None, control_freq=20, use_initial=False,
            initial_frame=0, dataset_cam_only=False, cam_distance=1.0):

    cam_pos, cam_quat = compute_dataset_cam(pos_cam_raw, scale, center, R, use_initial=use_initial, cam_distance=cam_distance)
    dataset_cam_key   = f"dataset_cam_{angle:g}_elev{elevation:g}_{eef_dir}"

    env = suite.make(
        env_name="Lift", robots="PandaRotatedGripper",
        gripper_types="Robotiq85Gripper",
        controller_configs=_load_ctrl_cfg(),
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=False,
        camera_names=CAMERAS, camera_heights=cam_h, camera_widths=cam_w,
        ignore_done=True, horizon=steps_per_waypoint * (len(pos) - initial_frame) + 100,
        hard_reset=False, control_freq=control_freq,
    )

    # Inject dataset camera and object mesh, then recompile
    root = ET.fromstring(env.model.get_xml())
    _set_offscreen_framebuffer(root, cam_w, cam_h)
    _hide_default_lift_props(root)
    _set_white_background(root)
    _hide_robot_mount_and_floor(root)
    worldbody = root.find('worldbody')
    ET.SubElement(worldbody, 'camera', {
        'name': DATASET_CAM,
        'pos':  ' '.join(f"{v:.4f}" for v in cam_pos),
        'quat': ' '.join(f"{v:.6f}" for v in cam_quat),
        'fovy': f'{fovy:.4f}',
    })
    ET.SubElement(worldbody, 'camera', SIDEVIEW_OPP)
    if data_dir is not None:
        inject_object_xml(root, data_dir)
    env._initialize_sim(ET.tostring(root, encoding='unicode'))
    env.reset()
    _set_droid_init_qpos(env)
    obs = env._get_observations()

    # Read EEF state directly from MuJoCo (obs may be stale after _set_droid_init_qpos)
    _site_id = next(iter(env.robots[0].eef_site_id.values())) if isinstance(env.robots[0].eef_site_id, dict) else int(env.robots[0].eef_site_id)

    # Move to the first waypoint through the controller, like deploy_franka.
    rot_eef_init      = Rotation.from_matrix(env.sim.data.site_xmat[_site_id].reshape(3, 3))
    rot_dataset_first = Rotation.from_quat(quat[initial_frame])
    rot_first_cmd = _parse_eef_dir(eef_dir) * rot_eef_init

    obs = move_to_start_pose(env, pos[initial_frame], rot_first_cmd)

    # Read site orientation directly from sim data after the startup move.
    rot_first = Rotation.from_matrix(env.sim.data.site_xmat[_site_id].reshape(3, 3))
    C_inv     = rot_first.inv() * rot_dataset_first
    
    # Rigidly attach object to EEF.
    quat0_mesh = quat[initial_frame]
    if mesh_rot_offset is not None:
        quat0_mesh = (Rotation.from_quat(quat[initial_frame]) * mesh_rot_offset.inv()).as_quat()
    update_object = make_object_updater(env, quat0_mesh) if data_dir is not None else None
    if update_object:
        update_object()

    render_cams = [DATASET_CAM] if dataset_cam_only else CAMERAS + [DATASET_CAM]
    renderers, render_data, render_option, render_camera_ids = make_renderers(
        env, cam_w, cam_h, render_cams
    )
    cam_mat     = {cam: get_cam_matrices(env, cam, cam_h, cam_w) for cam in render_cams}
    extra_cams  = [] if dataset_cam_only else CAMERAS
    frames      = {cam: [] for cam in extra_cams + [dataset_cam_key]}
    eef_history = []

    for i in range(len(pos) - initial_frame):
        tgt_pos = pos[i + initial_frame]
        tgt_quat = quat[i + initial_frame]
        rot_target = Rotation.from_quat(tgt_quat) * rot_dataset_first.inv() * rot_first
        action = np.concatenate([tgt_pos, rot_target.as_rotvec(), [1.0]])
        for _ in range(steps_per_waypoint):
            obs, *_ = env.step(action)

            if update_object:
                update_object()

            eef_history.append(obs["robot0_eef_pos"].copy())
            eef_quat_vis = (Rotation.from_quat(obs["robot0_eef_quat_site"]) * C_inv).as_quat()

            ds_img = render_camera(
                renderers[DATASET_CAM], render_data, render_option,
                render_camera_ids[DATASET_CAM], warmup=True
            )
            if show_eef:
                draw_eef(ds_img, eef_history, eef_quat_vis, *cam_mat[DATASET_CAM])
            frames[dataset_cam_key].append(ds_img)

            for cam in extra_cams:
                img = render_camera(renderers[cam], render_data, render_option, render_camera_ids[cam])
                if show_eef:
                    draw_eef(img, eef_history, eef_quat_vis, *cam_mat[cam])
                frames[cam].append(img)


        print(f"[{i+1}/{len(pos) - initial_frame}] err={np.linalg.norm(obs['robot0_eef_pos'] - tgt_pos):.4f}")

    for renderer in renderers.values():
        renderer.close()
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
    cfg = _load_config()
    initial_frame = cfg.get("start_frame", 0)
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir",    nargs="?", default=cfg.get("data_dir", "data/035_power_drill_20200709_151335"))
    parser.add_argument("--scale",     type=float, default=cfg.get("scale", 1))
    parser.add_argument("--steps",     type=int,   default=cfg.get("steps", 2))
    parser.add_argument("--video-dir", default="videos")
    parser.add_argument("--project",   default="robosuite-eef-traj")
    parser.add_argument("--name",      default=None)
    parser.add_argument("--wandb",     action="store_true")
    parser.add_argument("--show-eef",  action="store_true", default=cfg.get("show_eef", False))
    parser.add_argument("--angle",     type=float, default=cfg.get("angle", 90))
    parser.add_argument("--elevation", type=float, default=cfg.get("elevation", 0.0),
                        help="vertical camera angle in degrees (0=horizontal, 90=top-down)")
    parser.add_argument("--eef-dir",   default=cfg.get("eef_dir", "mz"),
                        help="gripper approach: 'mz'/'py'/'my' or 'SPH_lat<a>lon<b>[z<c>]' (e.g. 'SPH_lat180lon0' for top-down)")
    parser.add_argument("--ref-dir",   default=None,
                        help="reference data dir (e.g. data/ours) whose frame 0 defines the canonical mesh orientation")
    parser.add_argument("--control-freq", type=float, default=cfg.get("control_freq", 10),
                        help="OSC control frequency in Hz (default: 20)")
    parser.add_argument("--initial",     action="store_true", default=cfg.get("initial", False),
                        help="anchor traj start (frame 0) to center instead of mean")
    parser.add_argument("--dataset-cam-only", action="store_true", default=cfg.get("dataset_cam_only", False),
                        help="only render dataset_cam video (skip front/bird/sideview)")
    parser.add_argument("--cam-distance", type=float, default=cfg.get("cam_distance", 1.0),
                        help="dataset_cam distance multiplier (>1 = farther, <1 = closer)")
    args = parser.parse_args()

    _tmp_env = suite.make(
        env_name="Lift", robots="PandaRotatedGripper", gripper_types="Robotiq85Gripper",
        controller_configs=_load_ctrl_cfg(),
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=False, horizon=10,
    )
    _tmp_env.reset()
    _set_droid_init_qpos(_tmp_env)
    _tmp_robot = _tmp_env.robots[0]
    _tmp_site_id = next(iter(_tmp_robot.eef_site_id.values())) if isinstance(_tmp_robot.eef_site_id, dict) else int(_tmp_robot.eef_site_id)
    center = tuple(_tmp_env.sim.data.site_xpos[_tmp_site_id].copy())
    _tmp_env.close()
    
    center = tuple(np.array(center) + np.array(cfg.get("center_offset", [-0.1, 0.0, 0.1])))    
    
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    run_name  = args.name or data_dir.name
    R         = cam_to_robot_matrix(args.angle, args.elevation)

    pos_cam, quat_cam = load_traj(data_dir)
    pos, quat = cam_to_robot(pos_cam, quat_cam, R)
    pos, quat = remap(pos, quat, center=center, scale=args.scale, use_initial=args.initial)

    video_dir = Path(args.video_dir)
    if not video_dir.is_absolute():
        video_dir = PROJECT_ROOT / video_dir

    cam_json = json.load(open(data_dir / "camera.json"))
    cam_h    = cam_json["height"]
    cam_w    = cam_json["width"]
    fovy     = math.degrees(2 * math.atan(cam_h / (2 * cam_json["intrinsics"][1][1])))

    make_video(data_dir, show_eef=args.show_eef, start_frame=initial_frame)

    # mesh orientation correction: align frame-0 body frame to reference canonical pose
    mesh_rot_offset = None
    if args.ref_dir is not None:
        ref_dir = Path(args.ref_dir)
        if not ref_dir.is_absolute():
            ref_dir = PROJECT_ROOT / ref_dir
        R_ref  = np.load(ref_dir  / "object_pose" / "poses.npz")["poses"][0, :3, :3].astype(float)
        R_cur  = np.load(data_dir / "object_pose" / "poses.npz")["poses"][0, :3, :3].astype(float)
        mesh_rot_offset = Rotation.from_matrix(R_ref.T @ R_cur)

    wandb_run = wandb.init(project=args.project, name=run_name) if args.wandb else None
    run_sim(pos, quat, pos_cam, args.scale, center,
            args.steps, video_dir, run_name, wandb_run, R=R,
            angle=args.angle, elevation=args.elevation, eef_dir=args.eef_dir, show_eef=args.show_eef,
            fovy=fovy, cam_h=cam_h, cam_w=cam_w, data_dir=data_dir,
            mesh_rot_offset=mesh_rot_offset, control_freq=args.control_freq,
            use_initial=args.initial, initial_frame=initial_frame,
            dataset_cam_only=args.dataset_cam_only, cam_distance=args.cam_distance)
    if wandb_run:
        wandb_run.finish()
