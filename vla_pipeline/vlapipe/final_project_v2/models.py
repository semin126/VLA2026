"""Actual model adapters. No scripted policy fallback in A/B/C."""
import time


class ImagePlanner:
    def __init__(self, model_id, device='auto', max_new_tokens=1024):
        from ..planner import VLMPlanner
        self.backend = VLMPlanner(model_id=model_id, device=device)
        self.max_new_tokens = max_new_tokens

    def generate(self, image, prompt, deadline):
        from PIL import Image
        from transformers import StoppingCriteria, StoppingCriteriaList
        b = self.backend

        class Deadline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return time.perf_counter() >= deadline

        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': Image.fromarray(image)}, {'type': 'text', 'text': prompt}]}]
        kw = dict(add_generation_prompt=True, tokenize=True, return_dict=True)
        try:
            inputs = b.processor.apply_chat_template(messages, processor_kwargs={'return_tensors': 'pt'}, **kw)
            if not hasattr(inputs, 'to'):
                inputs = b.processor.apply_chat_template(messages, return_tensors='pt', **kw)
        except (TypeError, ValueError):
            inputs = b.processor.apply_chat_template(messages, return_tensors='pt', **kw)
        if time.perf_counter() >= deadline:
            return ''
        inputs = inputs.to(b.device, dtype=b.model.dtype)
        with b.torch.inference_mode():
            result = b.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      stopping_criteria=StoppingCriteriaList([Deadline()]))
        raw = b.processor.batch_decode(result[:, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)[0]
        return raw.strip()


def synchronize():
    import sys
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
