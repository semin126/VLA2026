"""Evaluate a base or SFT VLM on held-out collector images, without oracle fallback.

Scores unordered object/destination sets, not action order or robot success.
Only validation/test splits are accepted. A fresh output directory is required.
"""
import _bootstrap  # noqa: F401

import argparse
import json
import time
import statistics
from pathlib import Path

from PIL import Image

from finetune_planner import read_records
from vlapipe import config as C, numbered
from vlapipe.planner import VLMPlanner, parse_instruction, ZONE_ALIASES


def response_items(text, style):
    """Strict JSON schema; unknown IDs/zones count as false positives, never disappear."""
    text = text.strip()
    if text.startswith('```'):
        lines = text.splitlines()
        text = '\n'.join(lines[1:-1]).strip()
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError('Expected a JSON array')
    if style == 'number_read':
        if any(type(item) is not int for item in data):
            raise ValueError('number_read requires an array of integers')
        return set(data)
    result = set()
    for item in data:
        if not isinstance(item, dict) or not {'object', 'target_zone'} <= item.keys():
            raise ValueError('Each item requires object and target_zone')
        name, zone = str(item['object']), str(item['target_zone'])
        # This call normalizes the object identity only. The destination is scored
        # separately from the model's explicit target_zone below.
        mapped = parse_instruction(name, allow_default_zone=True)
        obj_key = mapped['object'] if mapped else 'unknown:' + name.casefold()
        zone_key = next((key for key, aliases in ZONE_ALIASES.items()
                         if zone.casefold() == key or zone.casefold() in aliases),
                        'unknown:' + zone.casefold())
        result.add((obj_key, zone_key))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--split', choices=['validation', 'test'], default='test')
    ap.add_argument('--model', default='HuggingFaceTB/SmolVLM2-500M-Video-Instruct')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    ap.add_argument('--max-new-tokens', type=int, default=800)
    args = ap.parse_args()
    manifest = json.loads((args.dataset / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete':
        raise SystemExit('Dataset is incomplete.')
    if args.max_new_tokens < 1:
        raise SystemExit('--max-new-tokens must be positive.')
    records = read_records(args.dataset / f'{args.split}.jsonl')
    style = manifest['prompt_style']
    if args.out.exists():
        raise SystemExit(f'Refusing to overwrite output: {args.out}')
    args.out.mkdir(parents=True)
    load_start = time.perf_counter()
    vlm = VLMPlanner(model_id=args.model, prompt_style=style, device=args.device)
    load_seconds = time.perf_counter() - load_start
    # Use stored prompts verbatim: do not silently change evaluation rules after collection.
    import torch
    import transformers
    if vlm.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    totals = {'tp': 0, 'fp': 0, 'fn': 0, 'exact': 0, 'invalid_json_or_schema': 0}
    elapsed = 0.0
    timings = []
    with (args.out / 'predictions.jsonl').open('x', encoding='utf-8') as stream:
        for row in records:
            if row['object_set'] == 'number':
                numbered.activate(count=row['object_count'])
            else:
                numbered.deactivate()
            with Image.open(row['image']) as source:
                image = source.convert('RGB')
            messages = [{'role': 'user', 'content': [
                {'type': 'image'}, {'type': 'text', 'text': row['prompt']}]}]
            template = vlm.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch = vlm.processor(text=template, images=[image], return_tensors='pt', add_special_tokens=False)
            batch = batch.to(vlm.device, dtype=vlm.model.dtype)
            if vlm.device.startswith('cuda'):
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                output = vlm.model.generate(**batch, max_new_tokens=args.max_new_tokens, do_sample=False)
            if vlm.device.startswith('cuda'):
                torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            raw = vlm.processor.batch_decode(output[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()
            gold = response_items(row['completion'], style)
            error = None
            try:
                pred = response_items(raw, style)
            except (ValueError, TypeError) as exc:
                pred = set()
                error = str(exc)
            tp, fp, fn = len(pred & gold), len(pred - gold), len(gold - pred)
            exact = error is None and pred == gold
            for key, value in {'tp': tp, 'fp': fp, 'fn': fn, 'exact': int(exact),
                               'invalid_json_or_schema': int(error is not None)}.items():
                totals[key] += value
            elapsed += seconds
            timings.append(seconds)
            stream.write(json.dumps({'id': row['id'], 'seed': row['seed'], 'layout': row['layout'],
                                     'raw_response': raw, 'gold': row['completion'],
                                     'tp': tp, 'fp': fp, 'fn': fn, 'exact': exact,
                                     'schema_error': error, 'generation_seconds': seconds,
                                     'oracle_fallback': False}, ensure_ascii=False) + '\n')
            stream.flush()
            print(row['id'], 'tp/fp/fn=', tp, fp, fn, 'exact=', exact, flush=True)
    tp, fp, fn = totals['tp'], totals['fp'], totals['fn']
    result = {'model': args.model, 'split': args.split, 'prompt_style': style,
              'n_examples': len(records), **totals,
              'precision_micro': tp / (tp + fp) if tp + fp else 0.0,
              'recall_micro': tp / (tp + fn) if tp + fn else 0.0,
              'f1_micro': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
              'exact_match': totals['exact'] / len(records),
              'generation_seconds_mean': elapsed / len(records),
              'generation_seconds_median': statistics.median(timings),
              'load_seconds': load_seconds,
              'device': vlm.device, 'dtype': str(vlm.model.dtype),
              'torch': torch.__version__, 'transformers': transformers.__version__,
              'model_revision': getattr(vlm.model.config, '_commit_hash', None),
              'max_new_tokens': args.max_new_tokens,
              'gpu_peak_allocated_gib': (torch.cuda.max_memory_allocated() / 2**30
                                         if vlm.device.startswith('cuda') else None),
              'note': 'Strict JSON unordered-set score; not physical success, accessibility sequencing, or a training benefit claim.'}
    (args.out / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
