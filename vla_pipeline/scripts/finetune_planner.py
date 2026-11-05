"""Small native Transformers SFT loop for collect_planner_dataset.py output.

Only assistant answer tokens contribute to loss. No TRL/PEFT dependency.
--dry-run validates the seed split and images without downloading a model.
Full fine-tuning is a research experiment; one optimizer step verifies plumbing,
not improved planning accuracy. Use a fresh --out for every run.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import random
from pathlib import Path


def read_records(path):
    with path.open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f'Empty dataset: {path}')
    from PIL import Image
    for row in rows:
        image = Path(row['image'])
        if not image.is_absolute():
            image = path.parent / image
        row['image'] = str(image.resolve())
        row['images'] = [row['image']]
        for message in row.get('messages', []):
            for block in message.get('content', []):
                if block.get('type') == 'image':
                    block['image'] = row['image']
        with Image.open(image) as im:
            im.verify()
        json.loads(row['completion'])
    return rows


def make_batch(processor, row, torch):
    """One image example, with an explicit checked prompt-token prefix mask."""
    from PIL import Image
    with Image.open(row['image']) as source:
        image = source.convert('RGB')
    user = [{'role': 'user', 'content': [
        {'type': 'image'}, {'type': 'text', 'text': row['prompt']}]}]
    full = user + [{'role': 'assistant', 'content': [
        {'type': 'text', 'text': row['completion']}]}]
    prompt_text = processor.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
    prefix = processor(text=prompt_text, images=[image], return_tensors='pt', add_special_tokens=False)
    batch = processor(text=full_text, images=[image], return_tensors='pt', add_special_tokens=False)
    n = prefix['input_ids'].shape[1]
    if not torch.equal(prefix['input_ids'][0], batch['input_ids'][0, :n]):
        raise ValueError('Chat template answer boundary is not an exact token prefix; refusing incorrect loss masking.')
    labels = batch['input_ids'].clone()
    labels[:, :n] = -100
    labels[batch['attention_mask'] == 0] = -100
    if int((labels != -100).sum()) == 0:
        raise ValueError('No assistant tokens to train.')
    batch['labels'] = labels
    return batch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', type=Path, required=True, help='Directory with complete manifest and train/validation/test.jsonl.')
    ap.add_argument('--model', default='HuggingFaceTB/SmolVLM2-500M-Video-Instruct')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--max-steps', type=int, default=100)
    ap.add_argument('--learning-rate', type=float, default=1e-5)
    ap.add_argument('--gradient-accumulation', type=int, default=1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    ap.add_argument('--autocast-bf16', action='store_true', help='Optional CUDA mixed precision; trainable parameters and optimizer stay FP32.')
    ap.add_argument('--gradient-checkpointing', action='store_true', help='Trade extra computation for lower activation memory on small GPUs.')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if args.max_steps < 1 or args.gradient_accumulation < 1:
        raise SystemExit('Step and accumulation counts must be positive.')
    manifest_path = args.dataset / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete':
        raise SystemExit('Dataset collection did not complete.')
    train = read_records(args.dataset / 'train.jsonl')
    validation = read_records(args.dataset / 'validation.jsonl')
    test = read_records(args.dataset / 'test.jsonl')
    groups = [set(row['seed'] for row in rows) for rows in (train, validation, test)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise SystemExit('Seed split overlap: refusing training leakage.')
    print(f'Validated train={len(train)}, validation={len(validation)}, test={len(test)}; seeds disjoint.')
    if args.dry_run:
        print('Dry run only: no model download, output creation or training.')
        return 0
    if args.out.exists():
        raise SystemExit(f'Refusing to overwrite output: {args.out}')
    import torch
    import transformers
    from transformers import AutoProcessor, AutoModelForImageTextToText
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    dtype = torch.float32  # Keep master parameters FP32 so small AdamW updates are not rounded away.
    if args.autocast_bf16 and (device != 'cuda' or not torch.cuda.is_bf16_supported()):
        raise SystemExit('--autocast-bf16 requires a CUDA device supporting BF16.')
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    args.out.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({'torch': torch.__version__, 'transformers': transformers.__version__,
                   'device': device, 'dtype': str(dtype),
                   'dataset_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                   'note': 'Full SFT; training loss is not a planning accuracy result. Test split is never optimized.'})
    (args.out / 'run_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    order = list(range(len(train)))
    random.shuffle(order)
    cursor = 0

    def batch_on_device(row):
        batch = make_batch(processor, row, torch)
        return {key: value.to(device=device, dtype=dtype) if value.is_floating_point()
                else value.to(device) for key, value in batch.items()}

    with (args.out / 'loss.jsonl').open('x', encoding='utf-8') as log:
        for step in range(1, args.max_steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for _ in range(args.gradient_accumulation):
                if cursor == len(order):
                    random.shuffle(order)
                    cursor = 0
                batch = batch_on_device(train[order[cursor]])
                cursor += 1
                with torch.autocast('cuda', dtype=torch.bfloat16) if args.autocast_bf16 else nullcontext():
                    loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite loss; stop before saving a misleading checkpoint.')
                losses.append(float(loss.detach()))
                (loss / args.gradient_accumulation).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            record = {'step': step, 'train_loss': sum(losses) / len(losses)}
            log.write(json.dumps(record) + '\n')
            log.flush()
            print(record, flush=True)
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16) if args.autocast_bf16 else nullcontext():
        val_losses = [float(model(**batch_on_device(row)).loss) for row in validation]
    (args.out / 'validation.json').write_text(json.dumps({
        'n_examples': len(validation), 'mean_example_loss': sum(val_losses) / len(val_losses),
        'note': 'Held-out language loss only; run planner accuracy and physical rollout separately.'}, indent=2), encoding='utf-8')
    model.save_pretrained(args.out / 'pretrained_model', safe_serialization=True)
    processor.save_pretrained(args.out / 'pretrained_model')
    print(f'Saved: {args.out / "pretrained_model"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
