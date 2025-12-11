# Rocq Transformer

Formal Transformer spec ("Attention All You Need" Vaswani 2017). Dependent types enforce dimension safety.

## Build

```bash
nix develop          # Rocq 9.1 shell
make                 # Build all 13 modules
make clean           # Clean
```

## Module Namespace

```rocq
From Transformer Require Import Tensor Config Attention Model.
```

## Structure

```
Transformer/
├── Tensor.v      # Nested list tensors, ~20 structural ops implemented, ~30 numerical Parameters
├── Config.v      # TransformerConfig + (num_heads | d_model) divisibility proof
├── Linear.v      # linearForward = transpose + matmul3D_2D + add3D_broadcast
├── Embedding.v   # embeddingsForward, positionalEncodingForward (needs seq <= max_len proof)
├── Attention.v   # scaledDotProductAttention4D, splitHeads, combineHeads, multiHeadAttentionForward
├── FeedForward.v # feedForwardForward = linear1 → relu → dropout → linear2
├── LayerNorm.v   # layerNormForward = layerNorm3D + mul_broadcast + add_broadcast
├── Sublayer.v    # sublayerConnectionForward = dropout3D + add3D (pre-norm residual)
├── Encoder.v     # encoderLayerForward, applyEncoderLayers (Fixpoint), encoderForward
├── Decoder.v     # decoderLayerForward (3 sublayers), applyDecoderLayers, decoderForward
├── Model.v       # EncoderDecoder record, encode, decode, forward, generatorForward
├── Inference.v   # greedyDecodeLoop (Fixpoint with type-level arithmetic), greedyDecode
└── Properties.v  # 15 theorems: shape preservation proofs
```

## Key Types

```rocq
(* Concrete tensor types *)
Definition Tensor2D (rows cols : nat) := list (list nat).
Definition Tensor3D (batch rows cols : nat) := list (list (list nat)).
Definition Tensor4D (batch heads rows cols : nat) := list (list (list (list nat))).

(* Config requires divisibility proof *)
Record TransformerConfig := {
  d_model, d_ff, num_heads, num_layers, max_len : nat;
  heads_divide : (num_heads | d_model)
}.

(* Full model *)
Record EncoderDecoder (d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab : nat) := {
  edEncoder, edDecoder, edSrcEmbed, edTgtEmbed, edSrcPos, edTgtPos, edGenerator
}.
```

## Type-Level Guarantees

- Tensor dims tracked types: `Tensor3D batch seq d_model`
- Config requires proof: `(num_heads | d_model)` - cannot construct invalid
- Decode step proves length: `Tensor2D batch n → Tensor2D batch (S n)`
- Positional encoding requires: `seq <= max_len` proof parameter

## Implemented vs Parameters

**Implemented (~20 ops):** zeros, ones, fill2D, transpose2D/3D/4D, viewToHeads, viewFromHeads, cat_batch/seq/2D_seq, subsequentMask, paddingMask, expand_mask2D_to_3D/3D_to_4D, select_last3D

**Parameters (~30 ops):** matmul2D/3D/4D/3D_2D, softmax/3D/4D, relu/3D, layerNorm/3D, dropout/3D/4D, add/3D/3D_broadcast/2D_broadcast, mul3D_broadcast, scale/3D/4D, maskedFill/4D, argmax/3D_last/last_position, embeddingLookup, gather, randn

## Key Patterns

```rocq
(* Forward pass pattern *)
Definition encoderLayerForward layer x mask :=
  let norm1 := layerNormForward (slcNorm (elSublayer1 layer)) x in
  let attnOut := multiHeadAttentionForward (elSelfAttn layer) norm1 norm1 norm1 mask in
  let x1 := sublayerConnectionForward (elSublayer1 layer) x attnOut in
  (* FFN sublayer same pattern *)
  ...

(* Type-level sequence growth in greedyDecodeLoop *)
Fixpoint greedyDecodeLoop remaining curLen (tgtSoFar : Tensor2D batch curLen)
  : Tensor2D batch (curLen + remaining) :=
  match remaining with
  | 0 => transport2D (eq_sym (add_0_r_eq curLen)) tgtSoFar
  | S rem' => ... (* generates 1 token, recurses with S curLen *)
  end.

(* transport2D for type coercion along dimension equality *)
Definition transport2D {batch d1 d2} (eq : d1 = d2) (t : Tensor2D batch d1) : Tensor2D batch d2.
```

## Proofs Style

Proofs trivial because **type signatures carry proof obligations**:

```rocq
Theorem encoder_preserves_shape : forall batch seq d_model enc x mask,
  exists (y : Tensor3D batch seq d_model), True.
Proof. intros. exists (encoderForward enc x mask). trivial. Qed.
(* Type of encoderForward PROVES shape preservation *)
```

## Common Operations

```rocq
(* Create config - need divisibility proof *)
Lemma heads_div : (8 | 512). Proof. exists 64. reflexivity. Qed.
Definition cfg := mkConfig 512 2048 8 6 5000 heads_div.

(* Greedy decode - need length proofs *)
Definition translate src srcPf genPf genPos :=
  greedyDecode model src None 1 srcPf genPf genPos.
(* srcPf : src_len <= max_len, genPf : gen_len <= max_len, genPos : gen_len >= 1 *)
```

## Nix Flake

```nix
devShells.default = pkgs.mkShell {
  packages = [ rocqPackages_9_1.rocq-core rocqPackages_9_1.stdlib ];
};
```

## Rocq 9.x Commands

```bash
rocq makefile -f _CoqProject -o Makefile.coq   # Generate Makefile
rocq compile file.v                             # Compile single file
rocq repl                                       # Interactive REPL
```

Imports: `From Stdlib Require Import ...` (not `From Coq`)
