"""
ProxDDP trajectory optimization for object pose trajectories.

Smooths a noisy SE(3) pose trajectory from freepose by solving a
constrained optimal control problem with the Franka Panda model:

  min  sum_t  w_d * ||log(T_t^{-1} T_ref_t)||^2
             + w_dq * ||dq_t||^2
             + w_tau * ||tau_t||^2

Usage (called as subprocess from run_batch.py / main_abs.py):
    <proxddp2_python> simulation/optimize_traj.py \
        <poses_npz_in> <poses_npz_out> \
        [--w-d 10] [--w-dq 0.01] [--w-tau 0.001] [--dt 0.1]
"""

import argparse
import numpy as np
import pinocchio as pin
import aligator
from aligator import dynamics, manifolds
from pathlib import Path
from scipy.spatial.transform import Rotation

# ── robot setup ───────────────────────────────────────────────────────────────

def _load_panda():
    import example_robot_data as erd
    robot = erd.load("panda")
    model = robot.model
    # Remove last two joints (finger joints) — keep only 7 arm joints
    # Actually keep them to avoid re-indexing; they're locked to 0 in control
    return model


def _ee_frame_id(model):
    for name in ("panda_hand", "panda_link8", "hand"):
        if model.existFrame(name):
            return model.getFrameId(name)
    raise RuntimeError("Could not find EE frame in panda model")


# ── trajectory optimization ───────────────────────────────────────────────────

def optimize(poses_cam: np.ndarray, w_d: float, w_dq: float, w_tau: float,
             dt: float) -> np.ndarray:
    """
    poses_cam: (N, 4, 4) float32 SE3 matrices in camera frame
    Returns:   (N, 4, 4) float64 smoothed SE3 matrices in camera frame
    """
    N = len(poses_cam)

    model = _load_panda()
    nq, nv = model.nq, model.nv
    nu = nv          # torque control, fully actuated
    ee_id = _ee_frame_id(model)

    space = manifolds.MultibodyPhaseSpace(model)
    ndx = space.ndx

    # neutral configuration as initial state
    q0 = pin.neutral(model)
    v0 = np.zeros(nv)
    x0 = np.concatenate([q0, v0])

    # forward kinematics at neutral to get EE pose
    data0 = model.createData()
    pin.framesForwardKinematics(model, data0, q0)
    T_ee0 = data0.oMf[ee_id].copy()

    # ── cam-frame poses → SE3 targets for EE ──────────────────────────────
    # We treat the cam-frame trajectory as desired Cartesian EE poses
    # (the optimization is in joint space; EE tracking drives the solution)
    T_cam0 = pin.SE3(poses_cam[0][:3, :3].astype(float),
                     poses_cam[0][:3, 3].astype(float))
    T_cam0_inv = T_cam0.inverse()

    # Build a list of SE3 targets: transform delta in cam frame → EE frame
    targets = []
    for i in range(N):
        T_cam_i = pin.SE3(poses_cam[i][:3, :3].astype(float),
                          poses_cam[i][:3, 3].astype(float))
        dT = T_cam0_inv * T_cam_i          # delta from first pose
        T_target = T_ee0 * dT              # apply delta to neutral EE pose
        targets.append(T_target)

    # ── dynamics ──────────────────────────────────────────────────────────
    ode  = dynamics.MultibodyFreeFwdDynamics(space)
    intg = dynamics.IntegratorSemiImplEuler(ode, dt)

    # ── stage costs ───────────────────────────────────────────────────────
    def make_stage(t):
        rcost = aligator.CostStack(space, nu)

        # EE placement tracking
        res_frame = aligator.FramePlacementResidual(ndx, nu, model, targets[t], ee_id)
        rcost.addCost("ee", aligator.QuadraticResidualCost(space, res_frame, w_d * np.eye(6)), 1.0)

        # joint velocity regularization
        res_v = aligator.StateErrorResidual(space, nu, x0)
        # velocity part is at indices nv:2*nv in the tangent space
        W_v = np.zeros(ndx)
        W_v[nv:2 * nv] = w_dq
        rcost.addCost("vel", aligator.QuadraticResidualCost(space, res_v, np.diag(W_v)), 1.0)

        # torque (control) regularization
        rcost.addCost("tau", aligator.QuadraticControlCost(space, nu, w_tau * np.eye(nu)), 1.0)

        return aligator.StageModel(rcost, intg)

    stages = [make_stage(t) for t in range(N - 1)]

    # terminal cost
    term_cost = aligator.CostStack(space, nu)
    res_term  = aligator.FramePlacementResidual(ndx, nu, model, targets[-1], ee_id)
    term_cost.addCost("ee_term", aligator.QuadraticResidualCost(space, res_term, w_d * np.eye(6)), 1.0)

    # ── problem + solver ──────────────────────────────────────────────────
    problem = aligator.TrajOptProblem(x0, stages, term_cost)

    solver = aligator.SolverProxDDP(1e-4, 1e-6)
    solver.max_iters = 500
    solver.verbose   = aligator.VerboseLevel.VERBOSE
    solver.setup(problem)

    # warm-start: repeat x0
    xs_init = [x0] * N
    us_init = [np.zeros(nu)] * (N - 1)
    solver.run(problem, xs_init, us_init)

    results = solver.results
    if not results.conv:
        print(f"[optimize_traj] WARNING: did not converge "
              f"(iters={results.num_iters}, prim_err={results.primal_infeas:.3e}, "
              f"dual_err={results.dual_infeas:.3e}). Using best-so-far solution.")
    xs_opt  = np.array(results.xs)   # (N, nq+nv)

    # ── reconstruct SE3 trajectory in cam frame ───────────────────────────
    poses_opt = np.zeros((N, 4, 4), dtype=np.float64)
    data_opt  = model.createData()
    for i, x_i in enumerate(xs_opt):
        q_i = x_i[:nq]
        pin.framesForwardKinematics(model, data_opt, q_i)
        T_ee_i = data_opt.oMf[ee_id]
        # invert the target mapping: cam_pose = T_cam0 * T_ee0^{-1} * T_ee_i
        dT_i = T_ee0.inverse() * T_ee_i
        T_cam_i = T_cam0 * dT_i
        poses_opt[i, :3, :3] = T_cam_i.rotation
        poses_opt[i, :3,  3] = T_cam_i.translation
        poses_opt[i,  3,  3] = 1.0

    return poses_opt


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("poses_in",  help="Input poses.npz path")
    parser.add_argument("poses_out", help="Output poses_opt.npz path")
    parser.add_argument("--w-d",    type=float, default=10.0,   help="SE3 tracking weight")
    parser.add_argument("--w-dq",   type=float, default=0.01,   help="Joint velocity weight")
    parser.add_argument("--w-tau",  type=float, default=0.001,  help="Torque weight")
    parser.add_argument("--dt",     type=float, default=0.1,    help="Timestep (s)")
    args = parser.parse_args()

    npz = np.load(args.poses_in)
    key = next((k for k in ["poses", "poses_cam"] if k in npz.files), None)
    if key is None:
        raise KeyError(f"No pose key in {npz.files}")
    poses_in = npz[key].astype(np.float64)
    frames   = npz.get("frames", None)

    print(f"[optimize_traj] Input: {args.poses_in}  ({len(poses_in)} frames)")
    poses_opt = optimize(poses_in, args.w_d, args.w_dq, args.w_tau, args.dt)
    print(f"[optimize_traj] Optimization done.")

    out_path = Path(args.poses_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {key: poses_opt.astype(np.float32)}
    if frames is not None:
        save_kwargs["frames"] = frames
    np.savez(out_path, **save_kwargs)
    print(f"[optimize_traj] Saved -> {out_path}")


if __name__ == "__main__":
    main()
