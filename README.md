# The Annotated Transformer (Rocq)

A formally verified specification of the Transformer architecture from
["Attention is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017),
written in [Rocq](https://rocq-prover.org/) (formerly Coq) with dependent types.

**If this code compiles, your Transformer has no dimension bugs.**

## What This Project Proves

The type system enforces these critical invariants at compile time:

| Invariant | Paper Reference | How It's Enforced |
|-----------|----------------|-------------------|
| Q·K^T requires matching inner dimensions | Equation (1) | `matmul4D` type signature |
| Multi-head splits d_model evenly | Section 3.2.2 | `(num_heads \| d_model)` proof required |
| Residual connections match shapes | Section 5.4 | `add3D` requires identical types |
| Self-attention preserves `[batch, seq, d_model]` | Section 3.2.1 | Return type = input type |
| Each decode step adds exactly 1 token | Inference algorithm | `Tensor2D batch n → Tensor2D batch (S n)` |
| Sequence length ≤ max positional encoding | Section 3.5 | Proof parameter required |

## Building

```bash
nix develop                    # Enter environment with Rocq 9.1
make                           # Compile all 13 modules (~10 seconds)
make clean                     # Remove compiled files
```

## The Core Ideas

### 1. Dimension-Indexed Tensors

In PyTorch, tensor shapes are runtime attributes that can silently mismatch:

```python
x = torch.randn(32, 128, 512)    # Hope this is [batch, seq, d_model]
y = linear(x)                     # Did the dimensions align? We'll find out at runtime...
```

In Rocq, shapes live in the type, verified at compile time:

```coq
Definition x : Tensor3D 32 128 512 := ...
Definition y : Tensor3D 32 128 256 := linearForward projection x.
(* Type checker verifies: input [32,128,512] → output [32,128,256] *)
```

### 2. Proof-Carrying Configuration

The Transformer requires `d_model` divisible by `num_heads` (so `d_k = d_model / num_heads`
is an integer). In Python, invalid configurations cause cryptic reshape errors:

```python
config = TransformerConfig(d_model=512, num_heads=7)  # 512/7 = 73.14... oops!
# Much later: "RuntimeError: shape '[32, 128, 7, 73]' is invalid for input of size..."
```

In Rocq, you **cannot construct** an invalid configuration:

```coq
Record TransformerConfig := {
  d_model   : nat;
  num_heads : nat;
  heads_divide : (num_heads | d_model)   (* PROOF required at construction *)
}.

(* This works: 8 divides 512 *)
Definition goodConfig := mkConfig 512 2048 8 6 5000 (exists 64; reflexivity).

(* This fails to compile: 7 does not divide 512 *)
Definition badConfig := mkConfig 512 2048 7 6 5000 ???.  (* No proof exists! *)
```

### 3. Type-Level Sequence Arithmetic

Autoregressive generation grows the sequence by exactly one token per step.
In Python, off-by-one errors lurk:

```python
for i in range(max_len):
    next_token = decode_step(sequence)
    sequence = torch.cat([sequence, next_token], dim=1)
    # What's the length now? i+1? sequence.shape[1]? Hope we're tracking correctly...
```

In Rocq, the return type **proves** the length:

```coq
Fixpoint greedyDecodeLoop
  (remaining : nat) (curLen : nat)
  (tgtSoFar : Tensor2D batch curLen)
  : Tensor2D batch (curLen + remaining) :=   (* Type IS the proof! *)
  match remaining with
  | 0 => tgtSoFar                            (* curLen + 0 = curLen *)
  | S rem' =>
      let next := decodeStep ... in          (* [batch, 1] *)
      let extended := cat tgtSoFar next in   (* [batch, curLen + 1] *)
      greedyDecodeLoop rem' (S curLen) extended
  end.
```

The type checker verifies the arithmetic: if this compiles, the sequence length is correct.

## Module Guide

The modules mirror the paper's structure:

```
Transformer/
├── Tensor.v        ─── Dimension-indexed tensor types (Section 3)
├── Config.v        ─── Configuration with divisibility proof (Section 3.2.2)
│
├── Linear.v        ─┬─ Building blocks
├── Embedding.v      │  Token + positional embeddings (Section 3.4, 3.5)
├── LayerNorm.v      │  Layer normalization (Section 5.4)
├── FeedForward.v   ─┘  Position-wise FFN (Section 3.3)
│
├── Attention.v     ─── Scaled dot-product & multi-head attention (Section 3.2)
├── Sublayer.v      ─── Pre-norm residual connections (Section 5.4)
│
├── Encoder.v       ─┬─ Encoder/decoder stacks (Section 3.1)
├── Decoder.v       ─┘  N layers with self/cross attention
│
├── Model.v         ─── Complete encoder-decoder architecture
├── Inference.v     ─── Greedy decoding with type-level proofs
└── Properties.v    ─── 15 theorems proving architectural correctness
```

### Paper-to-Code Correspondence

| Paper Section | Rocq Module | Key Definition |
|---------------|-------------|----------------|
| 3.1 Encoder-Decoder | `Model.v:39` | `EncoderDecoder` record |
| 3.2.1 Scaled Dot-Product | `Attention.v:33` | `scaledDotProductAttention4D` |
| 3.2.2 Multi-Head Attention | `Attention.v:115` | `multiHeadAttentionForward` |
| 3.3 Position-wise FFN | `FeedForward.v:53` | `feedForwardForward` |
| 3.4 Embeddings | `Embedding.v:26` | `embeddingsForward` |
| 3.5 Positional Encoding | `Embedding.v:51` | `positionalEncodingForward` |
| 5.4 Residual + LayerNorm | `Sublayer.v:30` | `sublayerConnectionForward` |

### Equations Implemented

**Equation (1): Attention(Q, K, V) = softmax(QK^T / √d_k) V**

```coq
(* Attention.v *)
Definition scaledDotProductAttention4D query key value mask :=
  let scores := matmul4D query (transpose4D_23 key) in   (* QK^T *)
  let scaled := scale4D scores in                        (* / √d_k *)
  let masked := maskedFill4D mask scaled in              (* apply mask *)
  let weights := softmax4D masked in                     (* softmax *)
  matmul4D weights value.                                (* × V *)
```

**Equation (2): FFN(x) = max(0, xW₁ + b₁)W₂ + b₂**

```coq
(* FeedForward.v *)
Definition feedForwardForward ff x :=
  let hidden := linearForward (ffLinear1 ff) x in        (* xW₁ + b₁ *)
  let activated := relu3D hidden in                      (* max(0, ·) *)
  let dropped := dropout3D activated in
  linearForward (ffLinear2 ff) dropped.                  (* ·W₂ + b₂ *)
```

**Sublayer: x + Dropout(Sublayer(LayerNorm(x)))**

```coq
(* Sublayer.v *)
Definition applySublayer slc x sublayerFn :=
  let normalized := layerNormForward (slcNorm slc) x in
  let sublayerOut := sublayerFn normalized in
  add3D x (dropout3D sublayerOut).                       (* Residual! *)
```

## Design Philosophy

### What's Implemented vs. Axiomatized

This is a **specification**, not an executable implementation. We distinguish:

**Implemented (real Rocq code, ~20 operations):**
- Tensor creation: `zeros`, `ones`, `fill2D`, `subsequentMask`
- Structural ops: `transpose`, `reshape`, `viewToHeads`, `viewFromHeads`
- Concatenation: `cat_batch`, `cat_seq`, `cat2D_seq`
- Masking: `expand_mask2D_to_3D`, `expand_mask3D_to_4D`

**Axiomatized (Parameters with precise type signatures, ~30 operations):**
- Numerical: `matmul`, `softmax`, `relu`, `layerNorm`, `dropout`
- Element-wise: `add`, `mul`, `sub`, `scale`, `maskedFill`
- Lookup: `embeddingLookup`, `argmax`, `gather`

**Why this split?**

Structural operations (transpose, reshape) are independent of numeric type—they work
identically for float32, float64, or symbolic tensors. We implement these fully.

Numerical operations require floating-point semantics that vary by backend (BLAS, cuBLAS, etc.).
We axiomatize them with **precise type signatures that encode their dimensional behavior**:

```coq
Parameter matmul2D : forall (m n k : nat),
  Tensor2D m k -> Tensor2D k n -> Tensor2D m n.
(* The type IS the specification: (m×k) @ (k×n) = (m×n) *)
```

### Why Proofs Are "Trivial"

Most proofs in `Properties.v` look like:

```coq
Theorem encoder_preserves_shape : forall (batch seq d_model : nat) ...,
  exists (y : Tensor3D batch seq d_model), True.
Proof. intros. exists (encoderForward enc x mask). trivial. Qed.
```

This isn't laziness—it's **the point**. The type signature `encoderForward : ... → Tensor3D batch seq d_model`
already **proves** shape preservation. The theorem just documents this guarantee explicitly.

If a function type-checks, it respects dimensional constraints. **Compilation is verification.**

## What Bugs Does This Catch?

### Bug 1: Mismatched Residual Connection

```python
# Python: Silent broadcasting or runtime crash
x = torch.randn(32, 128, 512)
sublayer_out = torch.randn(32, 128, 256)  # Wrong dimension!
out = x + sublayer_out  # ???
```

```coq
(* Rocq: Won't compile *)
Definition broken (x : Tensor3D 32 128 512) (out : Tensor3D 32 128 256) :=
  add3D 32 128 512 x out.
(* Error: Expected Tensor3D 32 128 512, got Tensor3D 32 128 256 *)
```

### Bug 2: Invalid Head Configuration

```python
# Python: Builds successfully, crashes later
config = TransformerConfig(d_model=512, num_heads=7)  # 512/7 ≠ integer
```

```coq
(* Rocq: Cannot construct *)
Definition broken := mkConfig 512 2048 7 6 5000 ???.
(* No proof that (7 | 512) exists *)
```

### Bug 3: Cross-Attention Mask Shape

```python
# Python: Cryptic error deep in attention
decoder_hidden = torch.randn(32, 10, 512)
encoder_memory = torch.randn(32, 20, 512)
mask = torch.ones(32, 10, 10)  # Should be (32, 10, 20)!
```

```coq
(* Rocq: Type error at call site *)
Definition broken (mask : Tensor3D 32 10 10) :=
  multiHeadAttentionForward mha decoder_hidden encoder_memory encoder_memory (Some mask).
(* Error: Expected Tensor3D 32 10 20, got Tensor3D 32 10 10 *)
```

## Minimal Example

```coq
From Transformer Require Import Tensor Config Model Inference.

(* Step 1: Configuration with divisibility proof *)
Lemma my_heads_divide : (8 | 512).
Proof. exists 64. reflexivity. Qed.

Definition myConfig := mkConfig 512 2048 8 6 5000 my_heads_divide.

(* Step 2: Construct model *)
Definition myModel := initEncoderDecoder 512 2048 8 6 5000 10000 10000.

(* Step 3: Type-checked forward pass *)
Definition translate
  (src : Tensor2D 1 20)        (* 1 sentence, 20 tokens *)
  (srcPf : 20 <= 5000)         (* Proof: seq_len ≤ max_len *)
  (genPf : 50 <= 5000)         (* Proof: output_len ≤ max_len *)
  (genPos : 50 >= 1)           (* Proof: generate at least 1 token *)
  : Tensor2D 1 50 :=           (* Output: 1 sentence, exactly 50 tokens *)
  greedyDecode myModel src None 1 srcPf genPf genPos.

(* If this compiles:
   - Source length (20) is within bounds
   - Generation length (50) is within bounds
   - Output is EXACTLY [1, 50] — not 49, not 51, exactly 50 *)
```

## Properties Verified

The 15 theorems in `Properties.v` prove:

1. **Self-attention preserves shape**: `[b,n,d] → [b,n,d]`
2. **Cross-attention uses query length**: `[b,q,d] × [b,k,d] → [b,q,d]`
3. **FFN preserves shape** (despite 4x internal expansion)
4. **Sublayer connection preserves shape** (enables residual)
5. **Encoder layer/stack preserves shape**
6. **Decoder layer/stack preserves shape**
7. **Encode produces source-length memory**
8. **Decode produces target-length output**
9. **Forward pass: tokens → logits with correct shapes**
10. **Greedy decode produces exactly gen_len tokens**
11. **Linear transforms last dimension only**
12. **Linear layers compose when dimensions align**
13. **Embedding output has d_model features**
14. **Positional encoding requires seq ≤ max_len**
15. **Each decode step extends by exactly 1**

## Related Work

This formal specification corresponds to:

- **[hs-annotated-transformer](https://github.com/jwiegley/hs-annotated-transformer)**: Executable Haskell implementation with hmatrix
- **[The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)**: Original Python/PyTorch tutorial by Harvard NLP

The Rocq version captures constraints that are only documented in comments in other implementations.

## Future Directions

Potential extensions:

1. **Semantic axioms**: Prove softmax sums to 1, layer norm produces unit variance
2. **Extraction**: Generate executable Haskell/OCaml from structural operations
3. **Training correctness**: Formalize gradient computation and optimizer steps
4. **Quantization proofs**: Verify precision bounds when quantizing to int8

## License

MIT
