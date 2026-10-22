# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Dino Twist
#
# Two parallel-jaw grippers grasp a deformable rubber dinosaur by the head and
# the tail, lift it off the table, then counter-roll it about its own long axis
# while pulling slightly apart. The dinosaur is a 876-vertex / 3168-tet
# VBD soft body with the per-tet material data from the asset export; the
# textured appearance surface follows the physics particles through the
# exported ``visual_to_physics`` binding.
#
# Command: uv run -m newton.examples dino_twist
#
###########################################################################

from __future__ import annotations

from itertools import pairwise

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.multiphysics.dino_asset import load_dino_asset, visual_surface_normals

# Rest pose from the reference choreography: the exported mesh is glTF y-up with
# +z pointing toward the head, so this rotation makes +z_local the world +x
# (head forward) and +y_local the world up axis.
DINO_POSITION = (0.36, 0.0, 0.06980535185)
DINO_ROTATION = (0.5, 0.5, 0.5, 0.5)

# Parallel-jaw hands: (label, local forward band for the grasped particles, world
# x approach sign, finger half extents (along, across, vertical), palm offset).
HANDS = (
    ("head", (0.06, 0.09), 1.0, (0.014, 0.005, 0.018), 0.022),
    ("tail", (-0.0917, -0.05), -1.0, (0.012, 0.004, 0.012), 0.020),
)

# Choreography: (time [s], hand offset along the approach axis [m], lift [m],
# counter-roll about the dinosaur's long axis [deg], jaw closure in [0, 1]).
# The head hand rolls by +roll and the tail hand by -roll, so the body is wrung
# about its own long axis while the hands pull slightly apart.
KEYFRAMES = (
    (0.0, 0.0, 0.0, 0.0, 1.0),
    (0.4, 0.0, 0.0, 0.0, 1.0),
    (1.8, 0.0, 0.13, 0.0, 1.0),
    (2.3, 0.0, 0.13, 0.0, 1.0),
    (4.0, 0.016, 0.15, 85.0, 1.0),
    (4.7, 0.018, 0.15, 95.0, 1.0),
    (6.0, 0.020, 0.15, 100.0, 1.0),
)

PARAMS = {
    "fps": 60,
    "num_frames": 372,
    "sim_substeps": 12,
    "solver_iterations": 8,
    "shear_scale": 0.08,
    "bulk_scale": 0.02,
    "tet_damping": 1.0e3,
    "attachment_stiffness": 4.0e3,
    "attachment_damping": 2.0,
    "grasp_time": 0.05,
    "particle_radius": 0.0015,
    "soft_contact_ke": 1.0e4,
    "soft_contact_kd": 10.0,
    "soft_contact_mu": 1.0,
}


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def sample_track(t: float) -> tuple[float, float, float, float]:
    """Interpolate the choreography with smooth easing."""
    if t <= KEYFRAMES[0][0]:
        return KEYFRAMES[0][1:]
    for (t0, *a), (t1, *b) in pairwise(KEYFRAMES):
        if t <= t1:
            s = smoothstep((t - t0) / max(t1 - t0, 1.0e-6))
            return tuple(va + (vb - va) * s for va, vb in zip(a, b, strict=True))
    return KEYFRAMES[-1][1:]


def cluster_rotation(rest: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Best-fit rotation taking a particle cluster from its rest to its current shape."""
    rest_centered = rest - rest.mean(axis=0)
    current_centered = current - current.mean(axis=0)
    u, _, vt = np.linalg.svd(rest_centered.T @ current_centered)
    correction = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, correction]) @ u.T


def rotation_angle(rotation: np.ndarray) -> float:
    """Rotation angle [rad] of a rotation matrix."""
    return float(np.arccos(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)))


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.asset = load_dino_asset(args.asset_dir)

        self.fps = PARAMS["fps"]
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame = 0
        self.total_frames = args.num_frames

        self.position = np.array(DINO_POSITION, dtype=np.float32)
        self.rotation = wp.quat(*DINO_ROTATION)

        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.001
        self.dino_start = len(builder.particle_q)
        self._add_dino(builder)
        self.dino_particles = np.arange(self.dino_start, len(builder.particle_q), dtype=np.int32)
        self.hands = self._add_hands(builder)
        self._create_attachments(builder)
        builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(ke=1.0e5, kd=10.0, mu=0.9, has_particle_collision=True),
            label="table",
        )
        builder.color()

        self.model = builder.finalize()
        self.model.soft_contact_ke = PARAMS["soft_contact_ke"]
        self.model.soft_contact_kd = PARAMS["soft_contact_kd"]
        self.model.soft_contact_mu = PARAMS["soft_contact_mu"]
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self._write_hands(0.0)
        wp.copy(self.state_1.body_q, self.state_0.body_q)

        self.attachments_engaged = False

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=args.solver_iterations,
            rigid_body_particle_contact_buffer_size=2048,
            particle_enable_self_contact=True,
            particle_self_contact_margin=0.0025,
            particle_self_contact_gap=0.0005,
            particle_topological_contact_filter_threshold=1,
        )
        self.collision_pipeline = newton.CollisionPipeline(self.model, broad_phase="explicit", soft_contact_gap=0.0015)
        self.contacts = self.collision_pipeline.contacts()

        particles = self.state_0.particle_q.numpy()
        self.rest_particles = particles.copy()
        self.rest_visual_points = particles[self.asset.visual_vertex_to_particle].copy()

        self.viewer.set_model(self.model)
        self._set_camera()

    # ── model construction ───────────────────────────────────────────────

    def _add_dino(self, builder: newton.ModelBuilder) -> None:
        asset = self.asset
        # Soften the exported 5 MPa rubber so the twist is visible at this scale.
        builder.add_soft_mesh(
            pos=self.position,
            rot=self.rotation,
            scale=1.0,
            vel=(0.0, 0.0, 0.0),
            vertices=asset.vertices,
            indices=asset.tet_indices,
            density=float(asset.density[0]),
            k_mu=asset.k_mu * self.args.shear_scale,
            k_lambda=asset.k_lambda * self.args.bulk_scale,
            k_damp=PARAMS["tet_damping"],
            particle_radius=PARAMS["particle_radius"],
            opacity=0.0,
            label="dino",
        )

    def _world_rest(self) -> np.ndarray:
        asset = self.asset
        return np.asarray(
            [wp.quat_rotate(self.rotation, wp.vec3(*p)) + wp.vec3(*self.position) for p in asset.vertices],
            dtype=np.float32,
        )

    def _add_hands(self, builder: newton.ModelBuilder) -> list[dict]:
        """One kinematic hand per grip point: two jaw bodies plus a connecting palm."""
        rest_local = self.asset.vertices
        rest_world = self._world_rest()
        cfg = newton.ModelBuilder.ShapeConfig(ke=1.0e5, kd=10.0, mu=1.0, has_particle_collision=True)

        hands = []
        for label, band, approach_sign, finger_half, palm_offset in HANDS:
            selected = np.flatnonzero((rest_local[:, 2] >= band[0]) & (rest_local[:, 2] <= band[1]))
            center = rest_world[selected].mean(axis=0)
            half_width = 0.5 * float(np.ptp(rest_world[selected][:, 1]))
            closed_gap = max(half_width * 0.9, 0.008)
            open_gap = half_width + 0.03
            approach = np.array((approach_sign, 0.0, 0.0), dtype=np.float32)

            bodies = []
            for side, color in ((-1.0, (0.92, 0.55, 0.16)), (1.0, (0.18, 0.45, 0.75))):
                body = builder.add_body(
                    xform=wp.transform(
                        wp.vec3(*(center + side * closed_gap * np.array((0.0, 1.0, 0.0)))), wp.quat_identity()
                    ),
                    mass=0.0,
                    is_kinematic=True,
                    label=f"gripper_{label}_{'left' if side < 0 else 'right'}",
                )
                builder.add_shape_box(
                    body, hx=finger_half[0], hy=finger_half[1], hz=finger_half[2], cfg=cfg, color=color
                )
                bodies.append(body)

            palm = builder.add_body(
                xform=wp.transform(wp.vec3(*(center + palm_offset * approach)), wp.quat_identity()),
                mass=0.0,
                is_kinematic=True,
                label=f"gripper_{label}_palm",
            )
            builder.add_shape_box(
                palm,
                hx=0.008,
                hy=closed_gap + max(finger_half[1], 0.012),
                hz=finger_half[2] * 0.8,
                cfg=cfg,
                color=(0.35, 0.36, 0.38),
            )
            bodies.append(palm)

            hands.append(
                {
                    "label": label,
                    "center": center,
                    "approach": approach,
                    "closed_gap": closed_gap,
                    "open_gap": open_gap,
                    "bodies": bodies,
                    "particles": selected,
                    "palm_offset": palm_offset,
                }
            )
        return hands

    def _hand_pose(self, hand: dict, t: float) -> dict[int, wp.transform]:
        offset, lift, roll, closed = sample_track(t)
        half_gap = hand["open_gap"] + (hand["closed_gap"] - hand["open_gap"]) * closed
        sign = 1.0 if hand["approach"][0] > 0.0 else -1.0
        center = hand["center"] + hand["approach"] * offset + np.array((0.0, 0.0, lift))
        # Counter-roll each hand about the dinosaur's long axis (world x at rest).
        rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(np.radians(roll) * sign))
        jaw_axis = np.array((0.0, 1.0, 0.0))

        poses = {}
        for side, body in zip((-1.0, 1.0), hand["bodies"][:2], strict=True):
            local = wp.quat_rotate(rot, wp.vec3(*(side * half_gap * jaw_axis)))
            poses[body] = wp.transform(wp.vec3(*center) + local, rot)
        palm = wp.quat_rotate(rot, wp.vec3(*(hand["palm_offset"] * hand["approach"])))
        poses[hand["bodies"][2]] = wp.transform(wp.vec3(*center) + palm, rot)
        return poses

    def _create_attachments(self, builder: newton.ModelBuilder) -> None:
        """Attach gripped particles to their nearest jaw with a compliant spring.

        Anchors are measured at the closed grasp pose, and all attachments start
        disabled so the approach is contact-only.
        """
        grasp_q = {}
        for hand in self.hands:
            grasp_q.update(self._hand_pose(hand, PARAMS["grasp_time"]))
        rest_world = self._world_rest()

        self.attachment_count = 0
        for hand in self.hands:
            jaws = hand["bodies"][:2]
            jaw_poses = {body: (np.array(grasp_q[body].p), wp.quat(*grasp_q[body].q)) for body in jaws}
            for particle in hand["particles"]:
                p = rest_world[particle]
                distances = [np.linalg.norm(p - pos) for pos, _ in jaw_poses.values()]
                body = jaws[int(np.argmin(distances))]
                if min(distances) > 0.035:
                    continue
                pos, rot = jaw_poses[body]
                local = wp.quat_rotate_inv(rot, wp.vec3(*p) - wp.vec3(*pos))
                builder.add_attachment_particle_body(
                    int(particle),
                    int(body),
                    body_point=local,
                    stiffness=PARAMS["attachment_stiffness"],
                    damping=PARAMS["attachment_damping"],
                    enabled=False,
                )
                self.attachment_count += 1

    # ── prescribed motion ────────────────────────────────────────────────

    def _write_hands(self, t: float) -> None:
        """Write hand poses and matching body velocities for the coming substeps."""
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        h = 1.0e-4
        for hand in self.hands:
            now = self._hand_pose(hand, t)
            nxt = self._hand_pose(hand, t + h)
            for body, xform in now.items():
                body_q[body] = np.array((*xform.p, *xform.q), dtype=np.float32)
                linear = (np.array(nxt[body].p, dtype=np.float32) - np.array(xform.p, dtype=np.float32)) / h
                q0 = wp.quat(*xform.q)
                dq = np.asarray(wp.normalize(wp.quat(*nxt[body].q) * wp.quat_inverse(q0)), dtype=np.float32)
                if dq[0] < 0.0:
                    dq = -dq
                angle = 2.0 * np.arctan2(np.linalg.norm(dq[1:]), max(float(dq[0]), 1.0e-9))
                axis = dq[1:] / max(float(np.linalg.norm(dq[1:])), 1.0e-9)
                body_qd[body] = np.array((*linear, *(axis * angle / h)), dtype=np.float32)
        self.state_0.body_q.assign(body_q)
        self.state_0.body_qd.assign(body_qd)

    def _engage_attachments(self) -> None:
        if self.attachments_engaged:
            return
        self.model.attachment_particle_body_enabled.assign(
            np.ones(self.model.attachment_particle_body_count, dtype=bool)
        )
        self.attachments_engaged = True

    # ── simulation loop ──────────────────────────────────────────────────

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.frame += 1
        t = self.frame * self.frame_dt
        self._write_hands(t)
        if t >= PARAMS["grasp_time"]:
            self._engage_attachments()
        self.simulate()
        self.sim_time += self.frame_dt
        self._compute_diagnostics()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self._log_dino_surface()
        self.viewer.end_frame()

    def _log_dino_surface(self) -> np.ndarray:
        particles = self.state_0.particle_q.numpy()
        points = particles[self.asset.visual_vertex_to_particle]
        normals = visual_surface_normals(points.astype(np.float64), self.asset.visual_faces)
        self.viewer.log_mesh(
            "dino_visual",
            wp.array(points, dtype=wp.vec3),
            wp.array(self.asset.visual_faces.flatten(), dtype=wp.int32),
            normals=wp.array(normals, dtype=wp.vec3),
            uvs=wp.array(self.asset.visual_uv, dtype=wp.vec2),
            texture=str(self.asset.albedo),
        )
        return points

    def _set_camera(self):
        target = np.array(DINO_POSITION) + np.array((0.0, 0.0, 0.11))
        eye = target + np.array((0.06, -0.46, 0.08))
        self.viewer.set_camera(wp.vec3(*eye))
        if hasattr(self.viewer, "camera"):
            self.viewer.camera.look_at(wp.vec3(*target))
            self.viewer.camera.fov = 40.0
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.exposure = 2.1

    # ── diagnostics ──────────────────────────────────────────────────────

    def _compute_diagnostics(self) -> None:
        particles = self.state_0.particle_q.numpy()
        rotations = []
        for hand in self.hands:
            idx = hand["particles"]
            rotations.append(cluster_rotation(self.rest_particles[idx], particles[idx]))
        # Counter-rotating ends differ by the total relative twist; a rigid body
        # translation or whole-body roll cancels out of this product.
        twist = rotation_angle(rotations[0] @ rotations[1].T)
        self.diagnostics = {
            "time": self.sim_time,
            "relative_twist_rad": float(twist),
            "min_particle_z": float(particles[:, 2].min()),
            "mean_particle_z": float(particles[:, 2].mean()),
            "extent": float(np.linalg.norm(np.ptp(particles, axis=0))),
        }

    def test_final(self):
        self._compute_diagnostics()
        if not np.isfinite(self.state_0.particle_q.numpy()).all():
            raise AssertionError("non-finite particle positions")
        if self.diagnostics["extent"] > 0.6:
            raise AssertionError(f"dinosaur exploded: extent={self.diagnostics['extent']:.3f}")
        if self.diagnostics["min_particle_z"] < -0.02:
            raise AssertionError(f"excessive table penetration: z_min={self.diagnostics['min_particle_z']:.4f}")
        if self.diagnostics["mean_particle_z"] < 0.12:
            raise AssertionError(f"dinosaur was not lifted: mean z={self.diagnostics['mean_particle_z']:.3f}")
        if abs(self.diagnostics["relative_twist_rad"]) < 0.3:
            raise AssertionError(f"dinosaur did not visibly twist: {self.diagnostics['relative_twist_rad']:.3f} rad")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=PARAMS["num_frames"])
        parser.add_argument("--asset-dir", type=str, default=None, help="Override the dino asset directory.")
        parser.add_argument("--substeps", type=int, default=PARAMS["sim_substeps"])
        parser.add_argument("--solver-iterations", type=int, default=PARAMS["solver_iterations"])
        parser.add_argument("--shear-scale", type=float, default=PARAMS["shear_scale"])
        parser.add_argument("--bulk-scale", type=float, default=PARAMS["bulk_scale"])
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
