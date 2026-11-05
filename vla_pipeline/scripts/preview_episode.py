"""Export recorded HDF5 camera observations as a review GIF, without simulation.

Movie playback is NOT physical action replay. Full episode by default; --stride
subsamples frames, while --max-frames explicitly truncates the review clip.
"""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path
import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('episode', type=Path)
    ap.add_argument('--camera', choices=['agentview', 'robot0_eye_in_hand'], default='agentview')
    ap.add_argument('--out', type=Path, required=True, help='New .gif path; no overwrite')
    ap.add_argument('--stride', type=int, default=2, help='Every Nth observation; preserves approximate playback speed')
    ap.add_argument('--max-frames', type=int, default=0, help='0=whole episode; positive value exports a truncated clip')
    args = ap.parse_args()
    if args.stride < 1 or args.max_frames < 0:
        ap.error('stride must be positive and max-frames nonnegative')
    if args.out.suffix.lower() != '.gif':
        ap.error('--out must end in .gif')
    sidecar = args.out.with_suffix('.preview.json')
    if args.out.exists() or sidecar.exists():
        raise FileExistsError('Choose a fresh GIF path; existing previews are preserved')
    key = f'obs/{args.camera}_image'
    with h5py.File(args.episode, 'r') as ep:
        if key not in ep:
            raise ValueError(f'No recorded camera {key}; this tool does not re-render missing views')
        frames = ep[key]
        if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8 or len(frames) == 0:
            raise ValueError(f'Expected nonempty uint8 THWC RGB frames, got {frames.shape}/{frames.dtype}')
        fps = float(ep.attrs['fps'])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f'Invalid recording fps: {fps}')
        indices = list(range(0, len(frames), args.stride))
        if indices[-1] != len(frames) - 1:
            indices.append(len(frames) - 1)
        truncated = bool(args.max_frames and len(indices) > args.max_frames)
        if args.max_frames:
            indices = indices[:args.max_frames]
        meta = json.loads(ep.attrs.get('meta', '{}'))
        report = {'source_episode': str(args.episode.resolve()), 'source_collection': meta.get('source'),
                  'task': str(ep.attrs.get('task', '')), 'direction': str(ep.attrs.get('direction', '')),
                  'camera': args.camera, 'source_fps': fps, 'stride': args.stride,
                  'source_observations': len(frames), 'source_actions': len(ep['actions']),
                  'source_action_seconds': len(ep['actions']) / fps,
                  'selected_observations': len(indices), 'first_observation': indices[0],
                  'last_observation': indices[-1], 'represented_end_seconds': indices[-1] / fps,
                  'truncated': truncated, 'contains_final_observation': indices[-1] == len(frames) - 1,
                  'physical_action_replay': False,
                  'note': 'Recorded-image movie only; no MuJoCo stepping or state forcing. '
                          'GIF timing is rounded; inspect the source for exact timestamps. '
                          'Duplicate image frames may be combined by the encoder.'}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        # Read one recorded frame at a time; do not load other cameras or the whole Episode.
        with imageio.get_writer(args.out, mode='I', duration=1000 * args.stride / fps, loop=0) as writer:
            for index in indices:
                writer.append_data(frames[index])
    with Image.open(args.out) as preview:
        report['encoded_gif_frames'] = preview.n_frames
        total_ms = 0
        for index in range(preview.n_frames):
            preview.seek(index)
            total_ms += preview.info.get('duration', 0)
        report['gif_duration_seconds'] = total_ms / 1000
    sidecar.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('GIF:', args.out)
    print('Review notes:', sidecar)


if __name__ == '__main__':
    main()
