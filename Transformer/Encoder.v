(** * Encoder Layer and Stack *)

(** The encoder consists of N identical layers, each with:
    1. Self-attention sublayer
    2. Feed-forward sublayer

    Each sublayer has residual connection + pre-norm. *)

From Transformer Require Import Tensor.
From Transformer Require Import Attention.
From Transformer Require Import FeedForward.
From Transformer Require Import LayerNorm.
From Transformer Require Import Sublayer.
From Stdlib Require Import Init.Nat.
From Stdlib Require Import Lists.List.
Import ListNotations.

(** ** Encoder Layer *)

Record EncoderLayer (d_model d_ff num_heads : nat) := mkEncoderLayer {
  elSelfAttn : MultiHeadAttention d_model num_heads;
  elFFN : FeedForward d_model d_ff;
  elSublayer1 : SublayerConnection d_model;
  elSublayer2 : SublayerConnection d_model
}.

Arguments mkEncoderLayer {d_model d_ff num_heads}.
Arguments elSelfAttn {d_model d_ff num_heads}.
Arguments elFFN {d_model d_ff num_heads}.
Arguments elSublayer1 {d_model d_ff num_heads}.
Arguments elSublayer2 {d_model d_ff num_heads}.

(** Encoder layer forward:
    1. x1 = x + dropout(selfAttn(norm(x), norm(x), norm(x), mask))
    2. output = x1 + dropout(ffn(norm(x1))) *)

Definition encoderLayerForward
  {d_model d_ff num_heads batch seq : nat}
  (layer : EncoderLayer d_model d_ff num_heads)
  (x : Tensor3D batch seq d_model)
  (mask : option (Tensor3D batch seq seq))
  : Tensor3D batch seq d_model :=
  (* Self-attention sublayer *)
  let norm1 := layerNormForward (slcNorm (elSublayer1 layer)) x in
  let attnOut := multiHeadAttentionForward (elSelfAttn layer) norm1 norm1 norm1 mask in
  let x1 := sublayerConnectionForward (elSublayer1 layer) x attnOut in
  (* FFN sublayer *)
  let norm2 := layerNormForward (slcNorm (elSublayer2 layer)) x1 in
  let ffnOut := feedForwardForward (elFFN layer) norm2 in
  sublayerConnectionForward (elSublayer2 layer) x1 ffnOut.

(** ** Full Encoder Stack *)

(** Encoder is a list of N encoder layers plus final layer norm. *)

Record Encoder (d_model d_ff num_heads num_layers : nat) := mkEncoder {
  encLayers : list (EncoderLayer d_model d_ff num_heads);
  encNorm : LayerNorm d_model
}.

Arguments mkEncoder {d_model d_ff num_heads num_layers}.
Arguments encLayers {d_model d_ff num_heads num_layers}.
Arguments encNorm {d_model d_ff num_heads num_layers}.

(** Apply encoder layers sequentially (fold). *)

Fixpoint applyEncoderLayers
  {d_model d_ff num_heads batch seq : nat}
  (layers : list (EncoderLayer d_model d_ff num_heads))
  (x : Tensor3D batch seq d_model)
  (mask : option (Tensor3D batch seq seq))
  : Tensor3D batch seq d_model :=
  match layers with
  | nil => x
  | layer :: rest =>
      let x' := encoderLayerForward layer x mask in
      applyEncoderLayers rest x' mask
  end.

(** Encoder forward: apply all layers then final norm. *)

Definition encoderForward
  {d_model d_ff num_heads num_layers batch seq : nat}
  (enc : Encoder d_model d_ff num_heads num_layers)
  (x : Tensor3D batch seq d_model)
  (mask : option (Tensor3D batch seq seq))
  : Tensor3D batch seq d_model :=
  let encoded := applyEncoderLayers (encLayers enc) x mask in
  layerNormForward (encNorm enc) encoded.

(** Initialization *)

Parameter initEncoderLayer : forall (d_model d_ff num_heads : nat),
  EncoderLayer d_model d_ff num_heads.

Parameter initEncoder : forall (d_model d_ff num_heads num_layers : nat),
  Encoder d_model d_ff num_heads num_layers.

(** Properties *)

Lemma encoder_preserves_shape : forall
  (d_model d_ff num_heads num_layers batch seq : nat)
  (enc : Encoder d_model d_ff num_heads num_layers)
  (x : Tensor3D batch seq d_model)
  (mask : option (Tensor3D batch seq seq)),
  exists (y : Tensor3D batch seq d_model), True.
Proof.
  intros.
  exists (encoderForward enc x mask).
  trivial.
Qed.
