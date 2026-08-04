"""Attaching GUARD to a live model without touching its weights or its forward pass."""

from guard.integration.torch_module import (
    GuardModule,
    candidate_embed_layers,
    reduce_embedding,
)

__all__ = ["GuardModule", "candidate_embed_layers", "reduce_embedding"]
