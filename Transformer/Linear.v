(** * Linear Layer for Transformer Model *)

(** This module defines the linear (fully-connected) layer with actual
    weight and bias tensors, implementing y = xW^T + b.

    The Linear layer is the fundamental building block for all projection
    layers in the transformer: Q, K, V, output projections in attention,
    and the feed-forward network layers. *)

From Transformer Require Import Tensor.
From Stdlib Require Import Init.Nat.

(** ** Linear Layer Type *)

(** A linear transformation layer with weight matrix and bias vector.

    - weight : [out_dim, in_dim] - the transformation matrix
    - bias : [out_dim] - the bias vector (represented as 2D for broadcasting)

    The transformation computes: y = xW^T + b *)

Record Linear (in_dim out_dim : nat) := mkLinear {
  linearWeight : Tensor2D out_dim in_dim;
  linearBias : Tensor2D 1 out_dim
}.

Arguments mkLinear {in_dim out_dim}.
Arguments linearWeight {in_dim out_dim}.
Arguments linearBias {in_dim out_dim}.

(** ** Forward Pass *)

(** Linear forward pass for 3D tensors (batched sequences).

    Computes y = xW^T + b where:
    - x : [batch, seq, in_dim]
    - W : [out_dim, in_dim]
    - W^T : [in_dim, out_dim]
    - xW^T : [batch, seq, out_dim]
    - b : [1, out_dim] (broadcast to [batch, seq, out_dim])
    - y : [batch, seq, out_dim]

    This directly implements the Haskell:
    linearForward layer input =
      let wT = transpose 0 1 (linearWeight layer)
          xW = matmul input wT
      in add xW (linearBias layer) *)

Definition linearForward {in_dim out_dim batch seq : nat}
  (layer : Linear in_dim out_dim)
  (input : Tensor3D batch seq in_dim)
  : Tensor3D batch seq out_dim :=
  let wT := transpose2D out_dim in_dim (linearWeight layer) in
  let xW := matmul3D_2D batch seq out_dim in_dim input wT in
  add3D_broadcast batch seq out_dim xW (linearBias layer).

(** Linear forward for 2D tensors (unbatched). *)

Definition linearForward2D {in_dim out_dim seq : nat}
  (layer : Linear in_dim out_dim)
  (input : Tensor2D seq in_dim)
  : Tensor2D seq out_dim :=
  let wT := transpose2D out_dim in_dim (linearWeight layer) in
  let xW := matmul2D seq out_dim in_dim input wT in
  add2D_broadcast seq out_dim xW (linearBias layer).

(** ** Initialization *)

(** Initialize a linear layer with random weights and zero bias.
    This is axiomatized since it requires random number generation. *)

Parameter initLinear : forall (in_dim out_dim : nat),
  Linear in_dim out_dim.

(** ** Properties *)

(** Composition: Two linear layers can be chained when dimensions align. *)

Lemma linear_compose : forall (d1 d2 d3 batch seq : nat)
  (l1 : Linear d1 d2) (l2 : Linear d2 d3)
  (x : Tensor3D batch seq d1),
  exists (y : Tensor3D batch seq d3), True.
Proof.
  intros.
  pose (intermediate := linearForward l1 x).
  pose (result := linearForward l2 intermediate).
  exists result.
  trivial.
Qed.

(** Q/K/V projections have identical type structure. *)

Lemma qkv_projections_same_type : forall (d_model batch seq : nat)
  (wq wk wv : Linear d_model d_model)
  (x : Tensor3D batch seq d_model),
  (exists (q : Tensor3D batch seq d_model), True) /\
  (exists (k : Tensor3D batch seq d_model), True) /\
  (exists (v : Tensor3D batch seq d_model), True).
Proof.
  intros.
  split; [|split].
  - exists (linearForward wq x). trivial.
  - exists (linearForward wk x). trivial.
  - exists (linearForward wv x). trivial.
Qed.
