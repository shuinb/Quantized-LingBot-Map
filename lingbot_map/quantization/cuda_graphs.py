from contextlib import contextmanager, ExitStack
import time

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten


@contextmanager
def cached_depth_position_embeddings(depth_head):
    """Reuse input-independent DPT encodings; warm before compile/capture."""
    from lingbot_map.heads.utils import create_uv_grid, position_grid_to_embed
    positions = {}
    stats = {'bytes': 0, 'entries': 0}
    previous = depth_head.__dict__.get('_apply_pos_embed')

    def cached_position(x, W, H, ratio=0.1):
        key = (x.shape[1], x.shape[-2], x.shape[-1], W, H, x.dtype, x.device, ratio)
        if key not in positions:
            grid = create_uv_grid(x.shape[-1], x.shape[-2], aspect_ratio=W / H,
                                  dtype=x.dtype, device=x.device)
            embedding = position_grid_to_embed(grid, x.shape[1]) * ratio
            positions[key] = embedding.permute(2, 0, 1)[None].contiguous()
            stats['bytes'] += embedding.numel() * embedding.element_size()
            stats['entries'] += 1
        return x + positions[key]

    depth_head._apply_pos_embed = cached_position
    try:
        yield stats
    finally:
        if previous is None:
            del depth_head._apply_pos_embed
        else:
            depth_head._apply_pos_embed = previous


class StatelessGraph:
    def __init__(self, forward, controller, index):
        self.forward = forward
        self.controller = controller
        self.graph = None
        self.signature = None
        self.static = None
        self.output = None
        self.stream = None
        self.index = index
        self.graph_forward = None

    def __call__(self, *args, **kwargs):
        tensors, structure = tree_flatten((args, kwargs))
        first = next((value for value in tensors if torch.is_tensor(value)), None)
        if not self.controller.enabled or not self.controller.allow_capture or first is None or first.shape[0] != 1:
            return self.forward(*args, **kwargs)
        if torch.is_grad_enabled() or not first.is_cuda:
            raise RuntimeError('Stateless CUDA Graphs require CUDA inference under no_grad')
        current_stream = (first.device, torch.cuda.current_stream(first.device).cuda_stream)
        if self.controller.share_pool:
            self.controller.check_order(self.index, first.device, current_stream)
        if self.stream is not None and self.stream != current_stream:
            raise RuntimeError('Stateless CUDA Graphs require one CUDA stream')
        signature = (structure, tuple((tuple(value.shape), value.dtype, value.device) if torch.is_tensor(value) else value for value in tensors),
                     torch.is_autocast_enabled('cuda'), torch.get_autocast_dtype('cuda'))
        if self.graph is not None and signature != self.signature:
            raise RuntimeError('Stateless CUDA Graph input changed; start a new graph context for a different shape or precision')
        if self.graph is None:
            self.stream = current_stream
            self.signature = signature
            self.static = [value.clone() if torch.is_tensor(value) else value for value in tensors]
            static_args, static_kwargs = tree_unflatten(self.static, structure)
            self.graph_forward = self.forward
            if self.controller.compile_stateless:
                started = time.perf_counter()
                self.graph_forward = torch.compile(self.forward, fullgraph=True, dynamic=False,
                                                   options={'triton.cudagraphs': False})
                # Complete compilation before entering CUDA Graph capture.
                # Joint prefill and disabled replay keep the original forward.
                self.graph_forward(*static_args, **static_kwargs)
                torch.cuda.synchronize(first.device)
                self.controller.compile_setup_seconds += time.perf_counter() - started
                self.controller.compilations += 1
            stream = self.controller.capture_stream if self.controller.share_pool else torch.cuda.Stream(device=first.device)
            stream.wait_stream(torch.cuda.current_stream(first.device))
            with torch.cuda.stream(stream), torch.autocast('cuda', enabled=torch.is_autocast_enabled('cuda'),
                                                          dtype=torch.get_autocast_dtype('cuda'), cache_enabled=False):
                for _ in range(3):
                    self.graph_forward(*static_args, **static_kwargs)
            torch.cuda.current_stream(first.device).wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream, pool=self.controller.pool), torch.autocast('cuda', enabled=torch.is_autocast_enabled('cuda'),
                                                                           dtype=torch.get_autocast_dtype('cuda'), cache_enabled=False):
                self.output = self.graph_forward(*static_args, **static_kwargs)
            self.controller.captures += 1
        for target, source in zip(self.static, tensors):
            if torch.is_tensor(source):
                target.copy_(source)
        self.graph.replay()
        outputs, output_structure = tree_flatten(self.output)
        result = tree_unflatten([value.clone() if torch.is_tensor(value) else value for value in outputs], output_structure)
        if self.controller.share_pool:
            self.controller.next_index = (self.index + 1) % self.controller.module_count
        return result


class GraphController:
    def __init__(self, share_pool=False, module_count=0, compile_stateless=False):
        self.enabled = True
        self.captures = 0
        self.allow_capture = True
        self.share_pool = share_pool
        self.module_count = module_count
        self.next_index = 0
        self.pool = None
        self.capture_stream = None
        self.replay_stream = None
        self.compile_stateless = compile_stateless
        self.compilations = 0
        self.compile_setup_seconds = 0.
        self.depth_position_cache = False
        self.position_cache_stats = {'bytes': 0, 'entries': 0}

    def check_order(self, index, device, stream):
        if index != self.next_index:
            raise RuntimeError('Shared graph pool requires configured stateless operations in order')
        if self.replay_stream is not None and stream != self.replay_stream:
            raise RuntimeError('Shared graph pool requires one CUDA device and stream')
        if self.pool is None:
            with torch.cuda.device(device):
                self.pool = torch.cuda.graph_pool_handle()
                self.capture_stream = torch.cuda.Stream(device=device)
            self.replay_stream = stream


@contextmanager
def stateless_cuda_graphs(model, share_pool=False, extended=False, compile_stateless=False):
    """Capture stateless work only; extended also covers SDPA QKV/FFN and DPT.

    Stateful attention, cache eviction and camera history always run eagerly.
    Prefill is never captured, including a single-anchor prefill after reset.
    Optional Inductor fusion is prepared before capture; it can change rounding.
    """
    if model.training:
        raise RuntimeError('Stateless CUDA Graphs require eval mode')
    if hasattr(model, '_stateless_graph_controller'):
        raise RuntimeError('Stateless CUDA Graphs already attached')
    aggregator = model.aggregator
    operations = [(aggregator.patch_embed, 'forward')]
    if extended:
        from lingbot_map.layers.block import SDPABlock
        if not all(isinstance(block, SDPABlock) for block in aggregator.global_blocks):
            raise ValueError('Extended stateless graphs currently require SDPA global blocks')
        # Follow the aggregator's actual alternating block schedule.
        for start in range(0, len(aggregator.frame_blocks), aggregator.aa_block_size):
            for kind in aggregator.aa_order:
                for index in range(start, start + aggregator.aa_block_size):
                    if kind == 'frame':
                        operations.append((aggregator.frame_blocks[index], 'forward'))
                    elif kind == 'global':
                        block = aggregator.global_blocks[index]
                        operations.extend(((block.attn, 'prepare_qkv'), (block, 'ffn_residual')))
        if getattr(model, 'depth_head', None) is not None:
            operations.append((model.depth_head, 'forward'))
    else:
        operations.extend((module, 'forward') for module in aggregator.frame_blocks)
    controller = GraphController(share_pool, len(operations), compile_stateless)
    model._stateless_graph_controller = controller
    restored = []
    contexts = ExitStack()
    handle = None
    if hasattr(model.aggregator, 'total_frames_processed'):
        def phase(module, inputs):
            controller.allow_capture = module.total_frames_processed > 0
        handle = model.aggregator.register_forward_pre_hook(phase)
    try:
        depth_head = getattr(model, 'depth_head', None)
        if extended and compile_stateless and hasattr(depth_head, '_apply_pos_embed'):
            # These values depend only on geometry/dtype, never on image content.
            # Warm eager prefill computes them outside graph-owned memory. Without
            # this reuse Inductor can inline FP64 pow/sin/cos at every output pixel.
            controller.position_cache_stats = contexts.enter_context(cached_depth_position_embeddings(depth_head))
            controller.depth_position_cache = True
        for index, (module, method) in enumerate(operations):
            previous = module.__dict__.get(method)
            restored.append((module, method, previous))
            graph = StatelessGraph(getattr(module, method), controller, index)
            if extended and module is getattr(model, 'depth_head', None):
                # DPT reads four feature levels and uses level 0 only for shape.
                # Do not copy all 24 feature maps into persistent graph inputs.
                used = set(module.intermediate_layer_idx) | {0}
                def depth_forward(tokens, *args, _graph=graph, _used=used, **kwargs):
                    selected = [tensor if i in _used else None for i, tensor in enumerate(tokens)]
                    return _graph(selected, *args, **kwargs)
                setattr(module, method, depth_forward)
            else:
                setattr(module, method, graph)
        yield controller
    finally:
        if handle is not None:
            handle.remove()
        for module, method, previous in restored:
            if previous is None:
                delattr(module, method)
            else:
                setattr(module, method, previous)
        contexts.close()
        del model._stateless_graph_controller
