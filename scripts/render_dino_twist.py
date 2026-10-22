# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Record the dino twist example with Newton's native textured OpenGL viewer.

The example itself is viewer agnostic; this script drives it headlessly, saves
every rendered frame and encodes an MP4. Frames and diagnostics land in
``results/dino_twist`` so partial runs are inspectable while tuning.

Run with::

    uv run --with imageio-ffmpeg python scripts/render_dino_twist.py

Add ``--no-video`` to only simulate, or ``--frames 180`` for a short preview.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.multiphysics.example_dino_twist import PARAMS, Example


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=None, help="Number of frames (default: example total).")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=str, default="results/dino_twist")
    parser.add_argument("--name", type=str, default="dino_twist")
    parser.add_argument("--substeps", type=int, default=20)
    parser.add_argument("--solver-iterations", type=int, default=12)
    parser.add_argument("--shear-scale", type=float, default=0.08)
    parser.add_argument("--bulk-scale", type=float, default=0.02)
    parser.add_argument("--asset-dir", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=1, help="Save every Nth rendered frame.")
    parser.add_argument("--no-video", action="store_true", help="Skip MP4 encoding.")
    parser.add_argument("--no-frames", action="store_true", help="Do not write PNG frames.")
    return parser.parse_args()


def example_args(args: argparse.Namespace) -> argparse.Namespace:
    """Subset of the example parser arguments used when driving the example directly."""
    return argparse.Namespace(
        asset_dir=args.asset_dir,
        substeps=args.substeps,
        solver_iterations=args.solver_iterations,
        shear_scale=args.shear_scale,
        bulk_scale=args.bulk_scale,
        num_frames=args.frames or PARAMS["num_frames"],
    )


def main() -> None:
    args = parse_args()
    wp.set_device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    if not args.no_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image  # noqa: PLC0415  (optional, only needed when writing frames)

    viewer = newton.viewer.ViewerGL(width=args.width, height=args.height, headless=True)
    example = Example(viewer, example_args(args))
    total = args.frames or example.total_frames
    print(f"device={viewer.device} frames={total} asset={example.asset.albedo.parent}")

    diagnostics = []
    recorded_points = []
    recorded_bodies = []
    started = time.perf_counter()
    for _ in range(total):
        example.step()
        example.render()
        diagnostics.append(dict(example.diagnostics))
        recorded_points.append(example.state_0.particle_q.numpy()[example.asset.visual_vertex_to_particle].copy())
        recorded_bodies.append(example.state_0.body_q.numpy().copy())
        if not args.no_frames and example.frame % args.save_every == 0:
            Image.fromarray(viewer.get_frame().numpy()).save(frames_dir / f"{example.frame:04d}.png")
        if example.frame % 30 == 0:
            elapsed = time.perf_counter() - started
            row = diagnostics[-1]
            print(
                f"frame {example.frame:4d}/{total} twist={row['relative_twist_rad']:+.3f} rad "
                f"z={row['mean_particle_z']:.3f} extent={row['extent']:.3f} "
                f"({elapsed / example.frame * 1e3:.1f} ms/frame)"
            )

    viewer.close()
    (output_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    np.savez_compressed(
        output_dir / f"{args.name}_recording.npz",
        visual_points=np.stack(recorded_points),
        body_q=np.stack(recorded_bodies),
        visual_faces=example.asset.visual_faces,
        visual_uv=example.asset.visual_uv,
        fps=example.fps,
    )
    print(f"simulated {total} frames in {time.perf_counter() - started:.1f}s")

    if args.no_frames or args.no_video:
        return

    import imageio_ffmpeg  # noqa: PLC0415  (optional, only needed when encoding)

    video_path = output_dir / f"{args.name}.mp4"
    frame_paths = sorted(frames_dir.glob("*.png"))
    frames = [np.asarray(Image.open(path).convert("RGB")) for path in frame_paths]
    height, width = frames[0].shape[:2]
    writer = imageio_ffmpeg.write_frames(str(video_path), (width, height), fps=example.fps, pix_fmt_in="rgb24")
    writer.send(None)
    for frame in frames:
        writer.send(frame.tobytes())
    writer.close()
    print(f"wrote {video_path} ({len(frames)} frames @ {example.fps} fps)")


if __name__ == "__main__":
    main()
