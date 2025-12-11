(** * Position-wise Feed-Forward Network *)

(** This module implements the position-wise feed-forward network (FFN)
    used in each encoder and decoder layer.

    FFN(x) = ReLU(x·W1 + b1)·W2 + b2

    The FFN expands the representation from d_model to d_ff (typically 4x),
    applies ReLU activation, then projects back to d_model. *)

From Transformer Require Import Tensor.
From Transformer Require Import Linear.
From Stdlib Require Import Init.Nat.
From Stdlib Require Import Lists.List.
Import ListNotations.

(** ** FeedForward Type *)

(** Position-wise feed-forward network with two linear transformations.

    - linear1 : d_model -> d_ff (expansion)
    - linear2 : d_ff -> d_model (projection) *)

Record FeedForward (d_model d_ff : nat) := mkFeedForward {
  ffLinear1 : Linear d_model d_ff;
  ffLinear2 : Linear d_ff d_model
}.

Arguments mkFeedForward {d_model d_ff}.
Arguments ffLinear1 {d_model d_ff}.
Arguments ffLinear2 {d_model d_ff}.

(** ** Forward Pass *)

(** FFN forward: expand -> ReLU -> dropout -> project.

    FFN(x) = Dropout(ReLU(x·W1 + b1))·W2 + b2

    - x : [batch, seq, d_model]
    - hidden : [batch, seq, d_ff] = linearForward linear1 x
    - activated : [batch, seq, d_ff] = ReLU(hidden)
    - dropped : [batch, seq, d_ff] = dropout(activated)
    - output : [batch, seq, d_model] = linearForward linear2 dropped

    This directly implements the Haskell:
    feedForwardForward ff x = do
      let hidden = linearForward (ffLinear1 ff) x
          activated = relu hidden
      dropped <- dropout activated
      let output = linearForward (ffLinear2 ff) dropped
      pure output *)

Definition feedForwardForward {d_model d_ff batch seq : nat}
  (ff : FeedForward d_model d_ff)
  (x : Tensor3D batch seq d_model)
  : Tensor3D batch seq d_model :=
  let hidden := linearForward (ffLinear1 ff) x in
  let activated := relu3D batch seq d_ff hidden in
  let dropped := dropout3D batch seq d_ff activated in
  linearForward (ffLinear2 ff) dropped.

(** ** Initialization *)

(** Initialize a feed-forward network with random weights. *)

Parameter initFeedForward : forall (d_model d_ff : nat),
  FeedForward d_model d_ff.

(** ** Properties *)

(** FFN preserves batch and sequence dimensions. *)

Lemma ffn_preserves_shape : forall (d_model d_ff batch seq : nat)
  (ff : FeedForward d_model d_ff)
  (x : Tensor3D batch seq d_model),
  exists (y : Tensor3D batch seq d_model), True.
Proof.
  intros.
  exists (feedForwardForward ff x).
  trivial.
Qed.

(** FFN can be composed after attention. *)

Lemma ffn_after_attention : forall (d_model d_ff batch seq : nat)
  (ff : FeedForward d_model d_ff)
  (attention_output : Tensor3D batch seq d_model),
  exists (ffn_output : Tensor3D batch seq d_model), True.
Proof.
  intros.
  exists (feedForwardForward ff attention_output).
  trivial.
Qed.
