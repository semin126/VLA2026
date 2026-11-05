"""Preserve checkpoint precision at inference; use FP32 master weights to train.

Transformers may construct different SmolVLA submodules in different dtypes.
Loading an FP32 checkpoint into BF16 parameters silently rounds its weights.
Set the destination parameter dtypes BEFORE safetensors copies the values.
"""
from collections import Counter

import torch
from safetensors import safe_open
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


class CheckpointPrecisionSmolVLA(SmolVLAPolicy):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # Complete checkpoints replace every parameter. Avoid initializing a
        # random backbone that will immediately be discarded; strict loading
        # prevents an incomplete file from leaving uninitialized parameters.
        from transformers.initialization import no_init_weights
        kwargs['strict'] = True
        with no_init_weights():
            return super().from_pretrained(*args, **kwargs)

    @classmethod
    def _load_as_safetensor(cls, model, model_file, map_location, strict):
        dtypes = {'F32': torch.float32, 'BF16': torch.bfloat16, 'F16': torch.float16,
                  'F64': torch.float64}
        with safe_open(model_file, framework='pt', device='cpu') as saved:
            current = model.state_dict(keep_vars=True)
            for name in saved.keys():
                tensor = current.get(name)
                dtype = dtypes.get(saved.get_slice(name).get_dtype())
                if tensor is not None and tensor.is_floating_point() and dtype is not None:
                    tensor.data = tensor.data.to(dtype=dtype)
        return super()._load_as_safetensor(model, model_file, map_location, strict)


class FloatMasterSmolVLA(SmolVLAPolicy):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.float()
        counts = Counter(str(p.dtype) for p in self.parameters() if p.requires_grad)
        if set(counts) != {'torch.float32'}:
            raise RuntimeError(f'Trainable master parameters must be FP32: {counts}')
        print(f'Trainable master parameter tensors: {dict(counts)}; AMP controls compute dtype', flush=True)


def install_float_master_factory():
    """Scope the native LeRobot factory override to this Python training process."""
    from lerobot.policies import factory
    original = factory.get_policy_class

    def get_policy_class(name):
        return FloatMasterSmolVLA if name == 'smolvla' else original(name)

    factory.get_policy_class = get_policy_class
