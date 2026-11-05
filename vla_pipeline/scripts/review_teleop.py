"""Record a human review in a NEW copy of a teleoperation demonstration.

Inspect the recording/scene first. This command records your decision, not an
automatic video judgement. Originals and rejected recordings are preserved.
"""
import _bootstrap  # noqa: F401
import argparse
from datetime import datetime
from pathlib import Path
from vlapipe.episode import Episode


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('episode', type=Path)
    ap.add_argument('--out', required=True, type=Path, help='New HDF5 file')
    ap.add_argument('--decision', choices=['accept', 'reject'], required=True)
    ap.add_argument('--reviewer', required=True)
    ap.add_argument('--notes', required=True)
    args = ap.parse_args()
    ep = Episode(args.episode)
    if args.out.exists():
        raise FileExistsError(args.out)
    if ep.meta.get('source') != 'human_teleoperation':
        raise ValueError('This review tool is for actual human teleoperation, not generated fixtures')
    if args.decision == 'accept' and ep.meta.get('success') is not True:
        raise ValueError('Mechanical screen failed; retain as failure and record a new demonstration')
    ep.meta.update({'review_status': 'accepted' if args.decision == 'accept' else 'rejected',
                    'reviewer': args.reviewer, 'review_notes': args.notes,
                    'reviewed_at': datetime.now().isoformat(timespec='seconds'),
                    'reviewed_from': str(args.episode.resolve())})
    if args.decision == 'reject':
        ep.meta['success'] = False
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ep.save(args.out)
    print(args.out)


if __name__ == '__main__':
    main()
