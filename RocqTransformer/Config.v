(** * Transformer Configuration with Dependent Type Constraints *)

(** This module defines the configuration parameters for the Transformer model
    with compile-time guarantees about dimensional constraints.

    The key constraint enforced at the type level is that d_model must be
    divisible by num_heads, ensuring that multi-head attention can split
    the model dimension evenly across all heads. *)

Require Import Coq.Init.Nat.
Require Import Coq.Arith.PeanoNat.
Require Import Coq.Arith.Arith.
Require Import Coq.micromega.Lia.

(** ** Divisibility Predicate *)

(** Definition of divisibility: a divides b if there exists k such that b = k * a *)
Definition Divides (a b : nat) : Prop :=
  exists k : nat, b = k * a.

(** Notation for divisibility *)
Notation "( a | b )" := (Divides a b) (at level 0).

(** ** Basic Divisibility Lemmas *)

(** Every number divides zero *)
Lemma divides_zero : forall a : nat,
  (a | 0).
Proof.
  intros a.
  exists 0.
  reflexivity.
Qed.

(** Every number divides itself *)
Lemma divides_refl : forall a : nat,
  a > 0 -> (a | a).
Proof.
  intros a Ha.
  exists 1.
  rewrite Nat.mul_1_l.
  reflexivity.
Qed.

(** Divisibility is transitive *)
Lemma divides_trans : forall a b c : nat,
  (a | b) -> (b | c) -> (a | c).
Proof.
  intros a b c [k1 Hk1] [k2 Hk2].
  exists (k2 * k1).
  rewrite Hk2, Hk1.
  rewrite Nat.mul_assoc.
  reflexivity.
Qed.

(** ** Training Mode *)

(** Training vs inference mode distinction *)
Inductive TrainingMode : Type :=
  | Training : TrainingMode
  | Inference : TrainingMode.

(** ** Training Environment *)

(** Training environment captures the execution mode.
    In the full implementation, this would also include dropout probability
    and other runtime configuration. *)
Record TrainingEnv : Type := mkTrainEnv {
  mode : TrainingMode
}.

(** ** Transformer Configuration with Dependent Constraint *)

(** Configuration record with compile-time guarantee that d_model is divisible
    by num_heads. This ensures that the head dimension (d_model / num_heads)
    is always a whole number.

    The fields mirror the Haskell TransformerConfig:
    - d_model: Model dimension (default: 512)
    - d_ff: Feed-forward dimension (default: 2048)
    - num_heads: Number of attention heads (default: 8)
    - num_layers: Number of encoder/decoder layers (default: 6)
    - max_len: Maximum sequence length (default: 5000)
    - heads_divide: Proof that num_heads divides d_model *)
Record TransformerConfig : Type := mkConfig {
  d_model : nat;
  d_ff : nat;
  num_heads : nat;
  num_layers : nat;
  max_len : nat;
  (* Critical constraint: d_model must be divisible by num_heads *)
  heads_divide : (num_heads | d_model)
}.

(** ** Head Dimension Computation *)

(** Compute the dimension of each attention head.
    This is guaranteed to be a whole number by the heads_divide constraint. *)
Definition head_dim (cfg : TransformerConfig) : nat :=
  d_model cfg / num_heads cfg.

(** ** Correctness Lemma for head_dim *)

(** Helper lemma: convert > 0 to <> 0 *)
Lemma gt_0_neq_0 : forall n : nat, n > 0 -> n <> 0.
Proof.
  intros n Hgt.
  lia.
Qed.

(** Proof that head_dim * num_heads = d_model when num_heads divides d_model
    and num_heads > 0. *)
Lemma head_dim_correct : forall cfg : TransformerConfig,
  num_heads cfg > 0 ->
  head_dim cfg * num_heads cfg = d_model cfg.
Proof.
  intros cfg Hnh_pos.
  unfold head_dim.
  destruct (heads_divide cfg) as [k Hk].
  rewrite Hk.
  (* Now we have: (k * num_heads cfg / num_heads cfg) * num_heads cfg = k * num_heads cfg *)
  rewrite Nat.div_mul by (apply gt_0_neq_0; exact Hnh_pos).
  (* Now we have: k * num_heads cfg = k * num_heads cfg *)
  reflexivity.
Qed.

(** Alternative formulation: d_model = k * num_heads for some k *)
Lemma d_model_factorization : forall cfg : TransformerConfig,
  exists k : nat, d_model cfg = k * num_heads cfg.
Proof.
  intros cfg.
  destruct (heads_divide cfg) as [k Hk].
  exists k.
  exact Hk.
Qed.

(** ** Default Configuration *)

(** Proof that the default configuration (512, 8) satisfies the divisibility constraint.
    512 = 64 * 8, so 8 divides 512. *)
Lemma default_heads_divide : (8 | 512).
Proof.
  exists 64.
  reflexivity.
Qed.

(** Default configuration matching the base model from "Attention is All You Need"
    (Vaswani et al., 2017).

    - Model dimension: 512
    - Feed-forward dimension: 2048 (4x model dimension)
    - Attention heads: 8
    - Encoder/decoder layers: 6
    - Maximum sequence length: 5000

    The configuration includes a proof that 8 divides 512 (head_dim = 64). *)
Definition defaultConfig : TransformerConfig := mkConfig
  512        (* d_model *)
  2048       (* d_ff = 4 * d_model *)
  8          (* num_heads *)
  6          (* num_layers *)
  5000       (* max_len *)
  default_heads_divide.

(** Verify that the default configuration has head dimension 64 *)
Example default_head_dim : head_dim defaultConfig = 64.
Proof.
  reflexivity.
Qed.

(** ** Smart Constructor for Valid Configurations *)

(** Create a valid configuration given parameters and a divisibility proof.
    This function requires the caller to provide a proof that num_heads divides d_model,
    ensuring the constraint is satisfied at construction time. *)
Definition mkValidConfig
  (dm df nh nl ml : nat)
  (pf : (nh | dm)) : TransformerConfig :=
  mkConfig dm df nh nl ml pf.

(** ** Example Configurations *)

(** Small configuration for testing or resource-constrained environments *)
Lemma small_heads_divide : (4 | 256).
Proof.
  exists 64.
  reflexivity.
Qed.

Definition smallConfig : TransformerConfig := mkConfig
  256        (* d_model *)
  1024       (* d_ff *)
  4          (* num_heads *)
  4          (* num_layers *)
  512        (* max_len *)
  small_heads_divide.

(** Large configuration for high-capacity models *)
Lemma large_heads_divide : (16 | 1024).
Proof.
  exists 64.
  reflexivity.
Qed.

Definition largeConfig : TransformerConfig := mkConfig
  1024       (* d_model *)
  4096       (* d_ff *)
  16         (* num_heads *)
  12         (* num_layers *)
  5000       (* max_len *)
  large_heads_divide.

(** ** Configuration Validation Properties *)

(** Helper lemma: if num_heads divides d_model and both are positive,
    then head_dim is positive *)
Lemma head_dim_positive : forall cfg : TransformerConfig,
  d_model cfg > 0 ->
  num_heads cfg > 0 ->
  head_dim cfg > 0.
Proof.
  intros cfg Hdm Hnh.
  unfold head_dim.
  destruct (heads_divide cfg) as [k Hk].
  rewrite Hk.
  (* k * num_heads cfg / num_heads cfg > 0 *)
  rewrite Nat.div_mul by (apply gt_0_neq_0; exact Hnh).
  (* k > 0 *)
  destruct k.
  - (* k = 0, contradiction with d_model > 0 *)
    rewrite Hk in Hdm. simpl in Hdm. lia.
  - (* k = S k', so S k' > 0 *)
    lia.
Qed.

(** The default configuration has all positive dimensions *)
Example default_config_valid :
  d_model defaultConfig > 0 /\
  d_ff defaultConfig > 0 /\
  num_heads defaultConfig > 0 /\
  num_layers defaultConfig > 0 /\
  head_dim defaultConfig > 0.
Proof.
  repeat split; unfold defaultConfig, head_dim; simpl; lia.
Qed.

(** ** Export Configuration *)

(** The configuration types and functions are now available for use in other modules.
    Key exports:
    - TransformerConfig: record type with dependent constraint
    - defaultConfig, smallConfig, largeConfig: example configurations
    - mkValidConfig: smart constructor requiring divisibility proof
    - head_dim: compute head dimension
    - head_dim_correct: proof that head_dim is correct
    - TrainingMode, TrainingEnv: training context types *)
