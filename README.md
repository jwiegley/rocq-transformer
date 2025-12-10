# Transformer

A machine-checked formal specification of the Transformer architecture in Rocq.

This project encodes the structural invariants of ["Attention is All You
Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) as dependent
types, providing compile-time guarantees that all tensor dimensions are correct
throughout the model.

## What This Proves

If this code compiles, then:

- **Matrix multiplications are compatible**: Q·Kᵀ requires matching inner dimensions
- **Residual connections match**: Input and sublayer output have identical shapes
- **Multi-head attention is well-formed**: d_model is provably divisible by num_heads
- **Positional encodings are bounded**: Sequence length never exceeds the encoding table
- **Autoregressive generation is sound**: Each decode step extends the sequence by exactly one token

## Building

```bash
nix develop   # Enter environment with Rocq 9.1
make          # Compile all 13 modules
```

## Architecture

The specification mirrors the paper's architecture:

```
Transformer/
├── Tensor.v      # Tensors indexed by dimension lists
├── Config.v      # Configuration with divisibility proof
├── Linear.v      # Linear projections
├── Embedding.v   # Token + positional embeddings
├── Attention.v   # Scaled dot-product & multi-head attention
├── FeedForward.v # Position-wise feed-forward network
├── LayerNorm.v   # Layer normalization
├── Sublayer.v    # Pre-norm residual connections
├── Encoder.v     # N-layer encoder stack
├── Decoder.v     # N-layer decoder with causal masking
├── Model.v       # Complete encoder-decoder
├── Inference.v   # Greedy decoding
└── Properties.v  # Cross-module correctness proofs
```

## Key Ideas

### Dimension-Indexed Tensors

```coq
Definition Tensor3D (batch seq dim : nat) := Tensor [batch; seq; dim].
```

A `Tensor3D 32 128 512` is a tensor of exactly that shape—not a runtime
property, but a type-level fact.

### Proof-Carrying Configuration

```coq
Record TransformerConfig := {
  d_model   : nat;
  num_heads : nat;
  heads_divide : (num_heads | d_model)   (* proof that num_heads divides d_model *)
}.
```

You cannot construct an invalid configuration. The divisibility witness is
required at construction time.

### Type-Level Sequence Growth

```coq
Parameter decodeStep :
  ... → Tensor2D batch n → Tensor2D batch (S n).
```

The successor `S n` in the return type *proves* that each step adds exactly one
token. This invariant is enforced by the type checker, not runtime assertions.

## Design Philosophy

This is an **abstract specification**, not an executable implementation.
Operations like `matmul` and `softmax` are declared as axioms (`Parameter`)
with precise type signatures. The implementation is left abstract because the
goal is to verify *structure*, not *computation*.

The proofs are intentionally simple—often just `reflexivity`—because the type
signatures themselves carry the proof obligations. If a function type-checks,
it respects the dimensional constraints.

## Modules

| Module | Purpose |
|--------|---------|
| `Tensor` | Abstract tensor type indexed by dimension list |
| `Config` | Transformer hyperparameters with d_model divisibility proof |
| `Linear` | Learned linear projection with in/out dimension tracking |
| `Embedding` | Token embeddings (scaled by √d_model) + sinusoidal positions |
| `Attention` | Scaled dot-product attention; multi-head with head split/combine |
| `FeedForward` | Two-layer FFN: d_model → d_ff → d_model |
| `LayerNorm` | Normalization preserving shape |
| `Sublayer` | Pre-norm residual: x + Dropout(Sublayer(LayerNorm(x))) |
| `Encoder` | Stack of (self-attention, FFN) layers |
| `Decoder` | Stack of (masked self-attn, cross-attn, FFN) layers |
| `Model` | Full encoder-decoder with embeddings and output projection |
| `Inference` | Greedy decode with type-level length increment |
| `Properties` | 35+ theorems proving architectural correctness |

## Related Work

This formal specification corresponds to the Haskell implementation at
[jwiegley/hs-annotated-transformer](https://github.com/jwiegley/hs-annotated-transformer),
which provides an executable reference. The Rocq version captures constraints
that are only documented in comments in the Haskell code.

## License

MIT
