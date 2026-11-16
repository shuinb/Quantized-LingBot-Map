"""Bound temporary GCT MLP activations during joint multi-frame initialization."""
from contextlib import contextmanager
from types import MethodType

import torch


def _chunked_mlp_forward(module, inputs):
    # Use the class method, not a captured bound-method default. Dynamo can
    # reuse this graph across identical blocks without guarding method IDs.
    forward = type(module).forward
    max_tokens = module._prefill_chunk_tokens
    rows = inputs.numel() // inputs.shape[-1]
    if rows <= max_tokens:
        return forward(module, inputs)
    flat = inputs.reshape(rows, inputs.shape[-1])
    first = forward(module, flat[:max_tokens])
    output = first.new_empty(rows, first.shape[-1])
    output[:max_tokens].copy_(first)
    del first
    for start in range(max_tokens, rows, max_tokens):
        output[start:start + max_tokens].copy_(forward(module, flat[start:start + max_tokens]))
    return output.reshape(*inputs.shape[:-1], output.shape[-1])


def _chunked_frame_forward(module, inputs, pos=None):
    forward = type(module).forward
    size = module._prefill_frame_batch_size
    if inputs.shape[0] <= size:
        return forward(module, inputs, pos=pos)
    output = torch.empty_like(inputs)
    for start in range(0, inputs.shape[0], size):
        positions = pos[start:start + size] if pos is not None and pos.shape[0] > 1 else pos
        output[start:start + size].copy_(forward(module, inputs[start:start + size], pos=positions))
    return output


@contextmanager
def chunked_prefill_mlp(model, max_tokens=0, *, frame_batch_size=0):
    """Evaluate independent token rows in chunks; keep joint attention intact.

    The returned tensor has the original shape. Streaming calls whose row count
    is already small keep their original forward, including CUDA Graph capture.
    An optional frame batch limit also splits independent per-frame attention.
    Global attention still attends jointly to every anchor. DINO and heads are
    unchanged.
    """
    if any(type(value) is not int or value < 0 for value in (max_tokens, frame_batch_size)) or not (max_tokens or frame_batch_size):
        raise ValueError('Use nonnegative chunk sizes with at least one positive size')
    if model.training or torch.is_grad_enabled():
        raise RuntimeError('Chunked prefill MLP requires eval inference under no_grad')
    restored = []
    try:
        modules = [(block.mlp, '_prefill_chunk_tokens', max_tokens, _chunked_mlp_forward)
                   for block in (*model.aggregator.frame_blocks, *model.aggregator.global_blocks)] if max_tokens else []
        if frame_batch_size:
            modules += [(block, '_prefill_frame_batch_size', frame_batch_size, _chunked_frame_forward)
                        for block in model.aggregator.frame_blocks]
        for module, attribute, limit, forward in modules:
            previous = module.__dict__.get('forward')
            if previous is not None:
                raise ValueError('Install prefill chunking before other forward overrides')
            previous_limit = module.__dict__.get(attribute)
            restored.append((module, attribute, previous_limit))
            setattr(module, attribute, limit)
            module.forward = MethodType(forward, module)
        yield
    finally:
        for module, attribute, previous_limit in restored:
            del module.forward
            if previous_limit is None:
                delattr(module, attribute)
            else:
                setattr(module, attribute, previous_limit)
