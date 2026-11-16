"""Optional single-frame camera graph with all keyframe history retained."""
from contextlib import contextmanager

import torch
from torch.nn import functional as F

from lingbot_map.heads.head_act import activate_pose
from lingbot_map.layers.rope import apply_rotary_emb


class CameraGraph:
    def __init__(self, head, reserve_frames):
        self.head = head
        self.forward = head.forward
        self.capacity = reserve_frames + 1  # One temporary non-keyframe slot.
        self.graph = None
        self.refresh = True
        self.captures = 0
        self.cache_bytes = 0
        self.signature = None

    def _step(self):
        head = self.head
        pose_tokens = head.token_norm(self.tokens)
        mask = (self.slots <= self.index).view(1, 1, 1, -1)
        prediction = None
        outputs = []
        for iteration in range(head.num_iterations):
            pose_input = head.empty_pose_tokens if prediction is None else prediction
            shift, scale, gate = head.poseLN_modulation(head.embed_pose(pose_input)).chunk(3, dim=-1)
            value = gate * (head.adaln_norm(pose_tokens) * (1 + scale) + shift) + pose_tokens
            for block, (keys, values) in zip(head.trunk, self.cache[iteration]):
                attention = block.attn
                normalized = block.norm1(value)
                qkv = attention.qkv(normalized).reshape(
                    1, 1, 3, attention.num_heads, attention.head_dim).permute(2, 0, 3, 1, 4)
                query, key, current_value = qkv.unbind(0)
                query, key = attention.q_norm(query), attention.k_norm(key)
                if self.position is not None:
                    query = apply_rotary_emb(query, self.position)
                    key = apply_rotary_emb(key, self.position)
                keys.index_copy_(2, self.index, key)
                values.index_copy_(2, self.index, current_value)
                attended = F.scaled_dot_product_attention(query, keys, values, attn_mask=mask, dropout_p=0.)
                if attention.gate_proj is not None:
                    score = attention.gate_proj(normalized).reshape(
                        1, 1, attention.num_heads, attention.head_dim).permute(0, 2, 1, 3)
                    attended = attended * torch.sigmoid(score)
                attended = attended.transpose(1, 2).reshape(1, 1, -1)
                value = value + block.ls1(attention.proj_drop(attention.proj(attended)))
                value = value + block.ls2(block.mlp(block.norm2(value)))
            delta = head.pose_branch(head.trunk_norm(value))
            prediction = delta if prediction is None else prediction + delta
            outputs.append(activate_pose(prediction, trans_act=head.trans_act,
                                         quat_act=head.quat_act, fl_act=head.fl_act))
        return outputs

    def __call__(self, aggregated_tokens_list, mask=None, num_iterations=None,
                 causal_inference=False, num_frame_per_block=1, **kwargs):
        head = self.head
        if not causal_inference:
            raise ValueError('Camera graph requires causal streaming inference')
        if head.frame_idx == 0:
            self.refresh = True
            return self.forward(aggregated_tokens_list, mask=mask, num_iterations=num_iterations,
                                causal_inference=True, num_frame_per_block=num_frame_per_block, **kwargs)
        tokens = aggregated_tokens_list[-1][:, :, 0]
        if (tokens.shape[:2] != (1, 1) or num_frame_per_block != 1 or mask is not None
                or num_iterations not in (None, head.num_iterations)):
            raise ValueError('Camera graph supports batch one, one frame, and the configured iteration count')
        if torch.is_grad_enabled() or head.training or not tokens.is_cuda:
            raise RuntimeError('Camera graph requires CUDA eval inference under no_grad')
        stream = torch.cuda.current_stream(tokens.device)
        signature = (tuple(tokens.shape), tokens.dtype, tokens.device, stream.cuda_stream,
                     torch.is_autocast_enabled('cuda'), torch.get_autocast_dtype('cuda'))
        if self.signature is not None and signature != self.signature:
            raise RuntimeError('Camera graph input precision, shape, device, or CUDA stream changed')
        if self.graph is None:
            self.signature = signature
            self.tokens = tokens.clone()
            self.index = torch.zeros(1, device=tokens.device, dtype=torch.int64)
            self.slots = torch.arange(self.capacity, device=tokens.device)
            self.cache = []
            for previous in head.kv_cache:
                layers = []
                for layer in range(head.trunk_depth):
                    key, value = previous[f'k_{layer}'], previous[f'v_{layer}']
                    if key.shape[3] != 1 or value.shape != key.shape:
                        raise ValueError('Camera graph expects one token per retained camera frame')
                    shape = (1, key.shape[1], self.capacity, key.shape[-1])
                    layers.append((torch.zeros(shape, device=key.device, dtype=key.dtype),
                                   torch.zeros(shape, device=value.device, dtype=value.dtype)))
                self.cache.append(layers)
            self.cache_bytes = sum(t.numel() * t.element_size() for layers in self.cache
                                   for pair in layers for t in pair)
        if self.refresh:
            self.count = head.kv_cache[0]['k_0'].shape[2]
            if self.count >= self.capacity:
                raise ValueError('Camera history exceeds reserved capacity')
            for previous, layers in zip(head.kv_cache, self.cache):
                for layer, (keys, values) in enumerate(layers):
                    key, value = previous[f'k_{layer}'], previous[f'v_{layer}']
                    if key.shape[2:4] != (self.count, 1):
                        raise ValueError('Camera cache layers have inconsistent history')
                    keys.zero_()
                    values.zero_()
                    keys[:, :, :self.count].copy_(key.flatten(2, 3))
                    values[:, :, :self.count].copy_(value.flatten(2, 3))
            self.refresh = False
        if self.count >= self.capacity:
            raise ValueError('Camera history exceeds reserved capacity; no frames were discarded')
        skip = head.kv_cache[0].get('_skip_append', False)
        if not skip and self.count >= self.capacity - 1:
            raise ValueError('Camera keyframe history exceeds reserved capacity; no frames were discarded')
        if any(previous.get('_skip_append', False) != skip for previous in head.kv_cache):
            raise ValueError('Camera iterations disagree on keyframe status')
        self.tokens.copy_(tokens)
        self.index.fill_(self.count)
        position = (head.rope3d(ppf=1, pph=1, ppw=1, patch_start_idx=0, device=tokens.device,
                                f_start=head.frame_idx, f_end=head.frame_idx + 1)
                    if head.rope3d is not None else None)
        if self.graph is None:
            self.position = position.clone() if position is not None else None
            capture_stream = torch.cuda.Stream(device=tokens.device)
            capture_stream.wait_stream(stream)
            with torch.cuda.stream(capture_stream), torch.autocast(
                    'cuda', enabled=torch.is_autocast_enabled('cuda'),
                    dtype=torch.get_autocast_dtype('cuda'), cache_enabled=False):
                for _ in range(3):
                    self._step()
            stream.wait_stream(capture_stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=capture_stream), torch.autocast(
                    'cuda', enabled=torch.is_autocast_enabled('cuda'),
                    dtype=torch.get_autocast_dtype('cuda'), cache_enabled=False):
                self.output = self._step()
            self.captures += 1
        elif self.position is not None:
            self.position.copy_(position)
        self.graph.replay()
        self.count += int(not skip)
        for previous, layers in zip(head.kv_cache, self.cache):
            for layer, (keys, values) in enumerate(layers):
                previous[f'k_{layer}'] = keys[:, :, :self.count].unsqueeze(3)
                previous[f'v_{layer}'] = values[:, :, :self.count].unsqueeze(3)
        head.frame_idx += 1
        return [value.clone() for value in self.output]


@contextmanager
def camera_cuda_graph(model, reserve_frames):
    """Capture camera math with padded, masked KV; retained history is unchanged.

    Padded SDPA can change floating-point rounding, so geometry must be evaluated.
    Joint prefill runs through the original head; a new sequence reloads its KV.
    """
    from lingbot_map.heads.camera_head import CameraCausalHead
    head = model.camera_head
    if not isinstance(head, CameraCausalHead) or head.training:
        raise ValueError('Camera graph requires an eval CameraCausalHead')
    if type(reserve_frames) is not int or reserve_frames < 1:
        raise ValueError('Camera graph requires a positive keyframe capacity')
    if any(block.attn.rope is not None for block in head.trunk):
        raise ValueError('Camera graph supports 3D rotary positions or no rotary positions')
    if hasattr(head, '_camera_graph_controller'):
        raise RuntimeError('Camera graph is already attached')
    controller = CameraGraph(head, reserve_frames)
    previous = head.__dict__.get('forward')
    head.forward = controller
    head._camera_graph_controller = controller
    try:
        yield controller
    finally:
        if previous is None:
            del head.forward
        else:
            head.forward = previous
        del head._camera_graph_controller
