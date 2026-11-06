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
    ap.add_argument('--save-every',type=int,default=0,help='Intermediate checkpoint interval; 0 saves at the end')
    ap.add_argument('--log-every',type=int,default=25)
    ap.add_argument('--mixed-precision',choices=['no','bf16','fp16'],default='no')
    ap.add_argument('--start-frame-weight',type=float,default=1.0,help='Train-only oversampling of early approach frames; 1 keeps native sampling')
    ap.add_argument('--start-frames',type=int,default=8)
    ap.add_argument('--frame-weights',type=Path,help='Optional train-only NPY weights indexed by original dataset frame; use identical aligned weights for matched A/B views')
    ap.add_argument('--train-vlm',action='store_true',help='Also adapt the VLA visual-language backbone; independent of the upper-level planner VLM')
    ap.add_argument('--train-vision',action='store_true',help='Also adapt the image encoder (implies --train-vlm)')
    ap.add_argument('--expert-only',action='store_true',help='Explicitly freeze the visual-language backbone, including when the base checkpoint previously trained it')
    ap.add_argument('--chunk-size',type=int,help='Optional predicted action horizon; independent of cached actions and the simulator subtask budget')
    ap.add_argument('--vision-cache',type=Path,help='Cache real frozen RGB encoder features for expert-only image training; disables image augmentation, inference remains native')
    ap.add_argument('--wait-for-gpu',action='store_true',help='Wait for this checkout model-training lock instead of rejecting a concurrent launch')
    ap.add_argument('--policy-action-steps','--action-steps',dest='policy_action_steps',type=int,default=10,
                    help='VLA cached actions before re-observing; distinct from the trial subtask horizon')
    args = ap.parse_args()
    if args.expert_only and (args.train_vlm or args.train_vision):
        ap.error('--expert-only cannot be combined with --train-vlm/--train-vision')
    if args.chunk_size is not None and args.chunk_size<1:
        ap.error('--chunk-size must be positive')
    episode_indices=[int(x) for x in args.episodes.split(',')] if args.episodes else None
    if episode_indices is not None and (not episode_indices or min(episode_indices)<0):
        ap.error('Episode indices must be nonnegative')
    if not (args.dataset_root / 'meta/info.json').is_file():
        ap.error('dataset-root must contain a converted train-only LeRobot dataset')
    if args.out.exists():
        ap.error('output already exists; choose a fresh path (no implicit overwrite/resume)')
    if args.steps < 1 or args.batch_size < 1 or args.save_every < 0 or args.log_every < 1 or args.start_frame_weight<1 or args.start_frames<1:
        ap.error('steps and batch-size must be positive')
    frame_sampling = None
    if args.frame_weights is not None:
        if args.start_frame_weight != 1:
            ap.error('--frame-weights and --start-frame-weight cannot be combined')
        import hashlib
        import io
        import json
        import numpy as np
        try:
            if args.frame_weights.suffix.lower() != '.npy':
                raise ValueError('Use a one-dimensional .npy array')
            frame_weights_bytes = args.frame_weights.read_bytes()
            frame_weights = np.load(io.BytesIO(frame_weights_bytes), allow_pickle=False)
            info = json.loads((args.dataset_root/'meta/info.json').read_text(encoding='utf-8-sig'))
            if (frame_weights.shape != (info['total_frames'],) or not np.isrealobj(frame_weights)
                    or not np.isfinite(frame_weights).all() or (frame_weights < 0).any()
                    or not (frame_weights > 0).any()):
                raise ValueError('Require one finite nonnegative weight per dataset frame, with a positive weight')
            frame_sampling = {'scheme':'physical_frame_weights',
                'replacement':True,'seed':args.seed,'dataset_root':str(args.dataset_root.resolve()),
                'source':str(args.frame_weights.resolve()),
                'sha256':hashlib.sha256(frame_weights_bytes).hexdigest(),
                'frames':len(frame_weights)}
        except (OSError, ValueError, TypeError, KeyError) as exc:
            ap.error(str(exc))
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.scripts import lerobot_train
    from vlapipe.model_loading import install_float_master_factory
    install_float_master_factory()
    if frame_sampling is not None:
        from functools import partial
        from vlapipe.training_sampling import FrameWeightedEpisodeSampler
        lerobot_train.EpisodeAwareSampler=partial(FrameWeightedEpisodeSampler,frame_weights=frame_weights)
        print('Train sampler: shared physical frame weights '+frame_sampling['sha256'],flush=True)
    if args.start_frame_weight!=1:
        from functools import partial
        from vlapipe.training_sampling import StartWeightedEpisodeSampler
        lerobot_train.EpisodeAwareSampler=partial(StartWeightedEpisodeSampler,
            start_frame_weight=args.start_frame_weight,start_frames=args.start_frames)
        print(f'Train sampler: first {args.start_frames} frames weighted {args.start_frame_weight}; validation remains unchanged',flush=True)

    policy = SmolVLAConfig.from_pretrained(args.base)
    bundled_metadata=Path(args.base)/'backbone_metadata'
    if bundled_metadata.is_dir():
        policy.vlm_model_name=str(bundled_metadata.resolve())
    if args.expert_only:
        policy.train_expert_only=True
        policy.freeze_vision_encoder=True
    if args.train_vlm or args.train_vision:
        policy.train_expert_only=False
        policy.freeze_vision_encoder=not args.train_vision
    # Complete SmolVLA checkpoints contain the backbone, too. The scoped factory
    # constructs all master parameters in FP32 BEFORE loading checkpoint values.
    policy.load_vlm_weights=False
    # A Hub id is not a filesystem path: Path() inserts backslashes on Windows.
    policy.pretrained_path = args.base
    policy.device = args.device
    policy.push_to_hub = False
    policy.optimizer_lr = args.learning_rate
    policy.scheduler_warmup_steps = min(args.warmup_steps, max(1, args.steps // 4))
    policy.scheduler_decay_steps = max(args.steps, policy.scheduler_warmup_steps + 1)
    if args.chunk_size is not None:
        policy.chunk_size=args.chunk_size
    policy.n_action_steps = args.policy_action_steps
    if not 1 <= policy.n_action_steps <= policy.chunk_size or args.learning_rate <= 0 or args.warmup_steps < 1:
        ap.error('Require positive learning rate/warmup and action-steps within chunk_size')
    policy.input_features = {}  # factory derives image keys + state16 from OUR dataset
    policy.output_features = {}  # factory derives action7, retaining pretrained padded head
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=args.repo_id, root=str(args.dataset_root.resolve()),episodes=episode_indices),
        policy=policy, output_dir=args.out.resolve(), job_name=args.out.name,
        steps=args.steps, batch_size=args.batch_size, num_workers=0,
        seed=args.seed, log_freq=args.log_every, save_freq=args.save_every or args.steps,
    )
    cfg.trackio.enable = False
    cfg.accelerator.mixed_precision=args.mixed_precision
    if args.vision_cache is not None:
        from vlapipe.training_vision_cache import install_frozen_vision_training
        cfg.dataset.image_transforms.enable=False
        install_frozen_vision_training(lerobot_train,cfg,args.vision_cache,compute=args.mixed_precision)
    import json
    from vlapipe.action_targets import descriptor
    representation_file=args.dataset_root/'action_representation.json'
    representation=json.loads(representation_file.read_text(encoding='utf-8-sig')) if representation_file.is_file() else descriptor()
    native_update=lerobot_train.update_last_checkpoint
    expected_root=args.out.resolve()/'checkpoints'
    def stamp_and_update(path):
        (Path(path)/'pretrained_model/action_representation.json').write_text(json.dumps(representation,indent=2),encoding='utf-8')
        if frame_sampling is not None:
            (Path(path)/'training_frame_sampling.json').write_text(json.dumps(frame_sampling,indent=2),encoding='utf-8')
            retained_weights=args.out/'frame_weights.npy'
            if not retained_weights.exists():
                retained_weights.write_bytes(frame_weights_bytes)
        if os.name=='nt':
            update_windows_last_checkpoint(path,expected_root)
        else:
            native_update(path)
    lerobot_train.update_last_checkpoint=stamp_and_update
    # LeRobot validate() re-reads sys.argv for special --policy.path flags.
    # Our argparse flags belong to this launcher, not LeRobot's parser.
    sys.argv = [sys.argv[0]]
    lerobot_train.train.__wrapped__(cfg)
    import json
    (args.out/'launch_options.json').write_text(json.dumps(vars(args),default=str,indent=2),encoding='utf-8')


if __name__ == '__main__':
    if '--help' in sys.argv or '-h' in sys.argv:
        main()  # Display usage even while another training process owns CUDA.
    from filelock import FileLock,Timeout
    lock_file=Path(__file__).resolve().parents[1]/'outputs/.vla_training.lock'
    lock_file.parent.mkdir(parents=True,exist_ok=True)
    training_lock=FileLock(str(lock_file))
    try:
        training_lock.acquire(timeout=-1 if '--wait-for-gpu' in sys.argv else 0)
    except Timeout as exc:
        raise SystemExit('Another VLA trainer is active in this checkout; preserve its output and wait.') from exc
    try:
        main()
    finally:
        training_lock.release()
