"""Fine-tune the pretrained SmolVLA backbone on this simulator's feature schema.

The pretrained robot's camera keys/state dimension must not survive unchanged.
Use a train-only dataset prepared by prepare_vla_training.py. A 1-step run tests
the software connection only; it is not a trained sorting policy.
"""
import _bootstrap  # noqa: F401
import argparse
import os
import sys
from pathlib import Path


def update_windows_last_checkpoint(checkpoint_dir: Path, expected_root: Path) -> None:
    """Use a directory junction; ordinary Windows accounts cannot create symlinks.

    Only the `last` link itself can be removed, never a checkpoint directory.
    Junctions have absolute targets; recreate this pointer after moving a run.
    """
    import _winapi

    root = expected_root.resolve(strict=True)
    target = Path(checkpoint_dir).resolve(strict=True)
    if target.parent != root or not target.is_dir() or target.name == 'last':
        raise ValueError(f'Checkpoint target is outside the expected run: {target}')
    last = root / 'last'
    if last.is_junction():
        if not last.resolve().is_relative_to(root):
            raise ValueError(f'Refusing to replace a last junction outside this run: {last}')
        last.rmdir()  # removes the junction only; does not traverse its target
    elif last.is_symlink():
        last.unlink()
    elif last.exists():
        raise FileExistsError(f'Refusing to remove a real checkpoint directory: {last}')
    _winapi.CreateJunction(str(target), str(last))
    if not last.is_junction() or last.resolve() != target:
        raise RuntimeError(f'Failed to verify last checkpoint junction: {last}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset-root', required=True, type=Path)
    ap.add_argument('--repo-id', required=True)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--base', default='lerobot/smolvla_base')
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=1000)
    ap.add_argument('--episodes',help='Optional comma-separated train dataset episode indices, for explicit overfit diagnostics')
    ap.add_argument('--learning-rate', type=float, default=1e-4)
    ap.add_argument('--warmup-steps', type=int, default=100)
    ap.add_argument('--policy-action-steps','--action-steps',dest='policy_action_steps',type=int,default=10,
                    help='VLA cached actions before re-observing; distinct from the trial subtask horizon')
    args = ap.parse_args()
    episode_indices=[int(x) for x in args.episodes.split(',')] if args.episodes else None
    if episode_indices is not None and (not episode_indices or min(episode_indices)<0):
        ap.error('Episode indices must be nonnegative')
    if not (args.dataset_root / 'meta/info.json').is_file():
        ap.error('dataset-root must contain a converted train-only LeRobot dataset')
    if args.out.exists():
        ap.error('output already exists; choose a fresh path (no implicit overwrite/resume)')
    if args.steps < 1 or args.batch_size < 1:
        ap.error('steps and batch-size must be positive')
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.scripts import lerobot_train

    policy = SmolVLAConfig.from_pretrained(args.base)
    # A Hub id is not a filesystem path: Path() inserts backslashes on Windows.
    policy.pretrained_path = args.base
    policy.device = args.device
    policy.push_to_hub = False
    policy.optimizer_lr = args.learning_rate
    policy.scheduler_warmup_steps = min(args.warmup_steps, max(1, args.steps // 4))
    policy.scheduler_decay_steps = max(args.steps, policy.scheduler_warmup_steps + 1)
    policy.n_action_steps = args.policy_action_steps
    if not 1 <= policy.n_action_steps <= policy.chunk_size or args.learning_rate <= 0 or args.warmup_steps < 1:
        ap.error('Require positive learning rate/warmup and action-steps within chunk_size')
    policy.input_features = {}  # factory derives image keys + state16 from OUR dataset
    policy.output_features = {}  # factory derives action7, retaining pretrained padded head
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=args.repo_id, root=str(args.dataset_root.resolve()),episodes=episode_indices),
        policy=policy, output_dir=args.out.resolve(), job_name=args.out.name,
        steps=args.steps, batch_size=args.batch_size, num_workers=0,
        seed=args.seed, log_freq=1, save_freq=args.steps,
    )
    cfg.trackio.enable = False
    if os.name == 'nt':
        expected_root = args.out.resolve() / 'checkpoints'
        # Override only this invocation; never modify installed LeRobot files.
        lerobot_train.update_last_checkpoint = lambda path: update_windows_last_checkpoint(path, expected_root)
    # LeRobot validate() re-reads sys.argv for special --policy.path flags.
    # Our argparse flags belong to this launcher, not LeRobot's parser.
    sys.argv = [sys.argv[0]]
    lerobot_train.train.__wrapped__(cfg)


if __name__ == '__main__':
    main()
