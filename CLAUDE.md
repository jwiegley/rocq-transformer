# Transformer (Rocq)

Formal specification of the Transformer architecture with compile-time dimension safety.

## Build

```bash
nix develop                    # Rocq 9.1
make                           # Build all modules
make Transformer/Tensor.vo     # Build specific module
```

## Module Namespace

```rocq
From Transformer Require Import Tensor Config Attention Model.
```

## Structure

```
Transformer/
├── Tensor.v      # Dimension-indexed tensor types
├── Config.v      # TransformerConfig with divisibility proof
├── Linear.v      # Linear projections
├── Embedding.v   # Token + positional embeddings
├── Attention.v   # Multi-head attention
├── FeedForward.v # Position-wise FFN
├── LayerNorm.v   # Layer normalization
├── Sublayer.v    # Pre-norm residual connections
├── Encoder.v     # N-layer encoder stack
├── Decoder.v     # N-layer decoder stack
├── Model.v       # Complete encoder-decoder
├── Inference.v   # Greedy decoding
└── Properties.v  # Cross-module proofs
```

## Type-Level Guarantees

- Tensor dimensions tracked in types: `Tensor3D batch seq d_model`
- Configuration requires divisibility proof: `(num_heads | d_model)`
- Decode step type proves length increment: `Tensor2D batch n → Tensor2D batch (S n)`

## Nix Flake

- `nix develop` — Development shell
- `nix build` — Build .vo files
- `nix flake check` — Verify compilation
