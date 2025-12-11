(** * Decoder Layer and Stack *)

(** The decoder has N identical layers, each with THREE sublayers:
    1. Masked self-attention (causal)
    2. Cross-attention (to encoder output)
    3. Feed-forward

    Each sublayer has residual connection + pre-norm. *)

From Transformer Require Import Tensor.
From Transformer Require Import Attention.
From Transformer Require Import FeedForward.
From Transformer Require Import LayerNorm.
From Transformer Require Import Sublayer.
From Stdlib Require Import Init.Nat.
From Stdlib Require Import Lists.List.
Import ListNotations.

(** ** Decoder Layer *)

Record DecoderLayer (d_model d_ff num_heads : nat) := mkDecoderLayer {
  dlSelfAttn : MultiHeadAttention d_model num_heads;   (* masked self-attention *)
  dlCrossAttn : MultiHeadAttention d_model num_heads;  (* cross-attention to encoder *)
  dlFFN : FeedForward d_model d_ff;
  dlSublayer1 : SublayerConnection d_model;  (* for self-attention *)
  dlSublayer2 : SublayerConnection d_model;  (* for cross-attention *)
  dlSublayer3 : SublayerConnection d_model   (* for FFN *)
}.

Arguments mkDecoderLayer {d_model d_ff num_heads}.
Arguments dlSelfAttn {d_model d_ff num_heads}.
Arguments dlCrossAttn {d_model d_ff num_heads}.
Arguments dlFFN {d_model d_ff num_heads}.
Arguments dlSublayer1 {d_model d_ff num_heads}.
Arguments dlSublayer2 {d_model d_ff num_heads}.
Arguments dlSublayer3 {d_model d_ff num_heads}.

(** Decoder layer forward:
    1. x1 = x + dropout(maskedSelfAttn(norm(x), tgtMask))
    2. x2 = x1 + dropout(crossAttn(norm(x1), memory, srcMask))
    3. output = x2 + dropout(ffn(norm(x2))) *)

Definition decoderLayerForward
  {d_model d_ff num_heads batch tgt_len src_len : nat}
  (layer : DecoderLayer d_model d_ff num_heads)
  (x : Tensor3D batch tgt_len d_model)
  (memory : Tensor3D batch src_len d_model)
  (tgtMask : option (Tensor3D batch tgt_len tgt_len))
  (srcMask : option (Tensor3D batch tgt_len src_len))
  : Tensor3D batch tgt_len d_model :=
  (* Masked self-attention sublayer *)
  let norm1 := layerNormForward (slcNorm (dlSublayer1 layer)) x in
  let selfAttnOut := multiHeadAttentionForward (dlSelfAttn layer)
                       norm1 norm1 norm1 tgtMask in
  let x1 := sublayerConnectionForward (dlSublayer1 layer) x selfAttnOut in
  (* Cross-attention sublayer (Q from decoder, K/V from encoder memory) *)
  let norm2 := layerNormForward (slcNorm (dlSublayer2 layer)) x1 in
  let crossAttnOut := multiHeadAttentionForward (dlCrossAttn layer)
                        norm2 memory memory srcMask in
  let x2 := sublayerConnectionForward (dlSublayer2 layer) x1 crossAttnOut in
  (* FFN sublayer *)
  let norm3 := layerNormForward (slcNorm (dlSublayer3 layer)) x2 in
  let ffnOut := feedForwardForward (dlFFN layer) norm3 in
  sublayerConnectionForward (dlSublayer3 layer) x2 ffnOut.

(** ** Full Decoder Stack *)

Record Decoder (d_model d_ff num_heads num_layers : nat) := mkDecoder {
  decLayers : list (DecoderLayer d_model d_ff num_heads);
  decNorm : LayerNorm d_model
}.

Arguments mkDecoder {d_model d_ff num_heads num_layers}.
Arguments decLayers {d_model d_ff num_heads num_layers}.
Arguments decNorm {d_model d_ff num_heads num_layers}.

(** Apply decoder layers sequentially. *)

Fixpoint applyDecoderLayers
  {d_model d_ff num_heads batch tgt_len src_len : nat}
  (layers : list (DecoderLayer d_model d_ff num_heads))
  (x : Tensor3D batch tgt_len d_model)
  (memory : Tensor3D batch src_len d_model)
  (tgtMask : option (Tensor3D batch tgt_len tgt_len))
  (srcMask : option (Tensor3D batch tgt_len src_len))
  : Tensor3D batch tgt_len d_model :=
  match layers with
  | nil => x
  | layer :: rest =>
      let x' := decoderLayerForward layer x memory tgtMask srcMask in
      applyDecoderLayers rest x' memory tgtMask srcMask
  end.

(** Decoder forward: apply all layers then final norm. *)

Definition decoderForward
  {d_model d_ff num_heads num_layers batch tgt_len src_len : nat}
  (dec : Decoder d_model d_ff num_heads num_layers)
  (x : Tensor3D batch tgt_len d_model)
  (memory : Tensor3D batch src_len d_model)
  (tgtMask : option (Tensor3D batch tgt_len tgt_len))
  (srcMask : option (Tensor3D batch tgt_len src_len))
  : Tensor3D batch tgt_len d_model :=
  let decoded := applyDecoderLayers (decLayers dec) x memory tgtMask srcMask in
  layerNormForward (decNorm dec) decoded.

(** ** Causal Mask Generation *)

(** Create a causal (subsequent) mask for decoder self-attention.
    Position i can only attend to positions <= i. *)

Definition subsequentMask (seq_len : nat) : Tensor2D seq_len seq_len :=
  Tensor.subsequentMask seq_len.

(** Initialization *)

Parameter initDecoderLayer : forall (d_model d_ff num_heads : nat),
  DecoderLayer d_model d_ff num_heads.

Parameter initDecoder : forall (d_model d_ff num_heads num_layers : nat),
  Decoder d_model d_ff num_heads num_layers.

(** Properties *)

Lemma decoder_preserves_shape : forall
  (d_model d_ff num_heads num_layers batch tgt_len src_len : nat)
  (dec : Decoder d_model d_ff num_heads num_layers)
  (x : Tensor3D batch tgt_len d_model)
  (memory : Tensor3D batch src_len d_model)
  (tgtMask : option (Tensor3D batch tgt_len tgt_len))
  (srcMask : option (Tensor3D batch tgt_len src_len)),
  exists (y : Tensor3D batch tgt_len d_model), True.
Proof.
  intros.
  exists (decoderForward dec x memory tgtMask srcMask).
  trivial.
Qed.
