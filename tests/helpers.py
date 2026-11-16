"""Small real streaming model for lifecycle tests; no pretrained checkpoint."""
from lingbot_map.aggregator.stream import AggregatorStream
from lingbot_map.models.gct_stream import GCTStream


class TinyGCT(GCTStream):
    def _build_aggregator(self):
        return AggregatorStream(img_size=28, patch_size=14, embed_dim=32, depth=24,
                                num_heads=4, patch_embed='conv', use_sdpa=True,
                                kv_cache_sliding_window=2, kv_cache_scale_frames=1,
                                use_gradient_checkpoint=False)
