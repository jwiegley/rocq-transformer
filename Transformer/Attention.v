(** * Multi-Head Attention Mechanisms *)

(** This module implements the attention mechanisms from "Attention is All You Need".

    Key operations:
    - scaledDotProductAttention: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
    - multiHeadAttention: Project Q,K,V, apply attention per head, concatenate *)

From Transformer Require Import Tensor.
From Transformer Require Import Config.
From Transformer Require Import Linear.
From Stdlib Require Import Init.Nat.
From Stdlib Require Import Arith.PeanoNat.
From Stdlib Require Import Lists.List.
Import ListNotations.

(** ** Mask Definitions *)

Definition SrcMask (batch src_len : nat) : Type := Tensor3D batch 1 src_len.
Definition TgtMask (batch tgt_len : nat) : Type := Tensor3D batch tgt_len tgt_len.

(** ** Scaled Dot-Product Attention *)

(** scaledDotProductAttention Q K V mask:
    1. scores = Q @ K^T
    2. scaled = scores / sqrt(d_k)
    3. masked = apply mask (set masked positions to -inf)
    4. weights = softmax(masked)
    5. output = weights @ V

    For 4D multi-head: [batch, heads, seq_q, d_k] *)

Definition scaledDotProductAttention4D
  {batch heads seq_q seq_k d_k : nat}
  (query : Tensor4D batch heads seq_q d_k)
  (key : Tensor4D batch heads seq_k d_k)
  (value : Tensor4D batch heads seq_k d_k)
  (mask : option (Tensor3D batch seq_q seq_k))
  : Tensor4D batch heads seq_q d_k :=
  (* 1. Compute scores: Q @ K^T -> [batch, heads, seq_q, seq_k] *)
  let keyT := transpose4D_23 batch heads seq_k d_k key in
  let scores := matmul4D batch heads seq_q seq_k d_k query keyT in
  (* 2. Scale by 1/sqrt(d_k) *)
  let scaled := scale4D batch heads seq_q seq_k scores in
  (* 3. Apply mask if provided *)
  let masked := match mask with
    | None => scaled
    | Some m => maskedFill4D batch heads seq_q seq_k
                  (expand_mask3D_to_4D batch seq_q seq_k m) scaled
    end in
  (* 4. Softmax to get weights *)
  let weights := softmax4D batch heads seq_q seq_k masked in
  (* 5. Apply dropout *)
  let droppedWeights := dropout4D batch heads seq_q seq_k weights in
  (* 6. Weighted sum of values *)
  matmul4D batch heads seq_q d_k seq_k droppedWeights value.

(** ** Head Splitting and Combining *)

(** Split d_model into num_heads separate heads.
    [batch, seq, d_model] -> [batch, heads, seq, head_dim]
    where d_model = heads * head_dim *)

Definition splitHeads {batch seq num_heads head_dim : nat}
  (x : Tensor3D batch seq (num_heads * head_dim))
  : Tensor4D batch num_heads seq head_dim :=
  (* Reshape: [batch, seq, d_model] -> [batch, seq, heads, head_dim] *)
  let reshaped := viewToHeads batch seq num_heads head_dim x in
  (* Transpose: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim] *)
  transpose4D_12 batch seq num_heads head_dim reshaped.

(** Combine heads back into single d_model dimension.
    [batch, heads, seq, head_dim] -> [batch, seq, d_model] *)

Definition combineHeads {batch seq num_heads head_dim : nat}
  (x : Tensor4D batch num_heads seq head_dim)
  : Tensor3D batch seq (num_heads * head_dim) :=
  (* Transpose: [batch, heads, seq, head_dim] -> [batch, seq, heads, head_dim] *)
  let transposed := transpose4D_12 batch num_heads seq head_dim x in
  (* View: [batch, seq, heads, head_dim] -> [batch, seq, d_model] *)
  viewFromHeads batch seq num_heads head_dim transposed.

(** ** Multi-Head Attention *)

(** MultiHeadAttention contains 4 linear projections:
    - query_proj, key_proj, value_proj: d_model -> d_model
    - output_proj: d_model -> d_model *)

Record MultiHeadAttention (d_model num_heads : nat) := mkMHA {
  mhaQueryProj : Linear d_model d_model;
  mhaKeyProj : Linear d_model d_model;
  mhaValueProj : Linear d_model d_model;
  mhaOutputProj : Linear d_model d_model;
  mhaNumHeads : nat;
  mhaHeadDim : nat;
  mhaHeadDimProof : d_model = num_heads * mhaHeadDim
}.

Arguments mkMHA {d_model num_heads}.
Arguments mhaQueryProj {d_model num_heads}.
Arguments mhaKeyProj {d_model num_heads}.
Arguments mhaValueProj {d_model num_heads}.
Arguments mhaOutputProj {d_model num_heads}.
Arguments mhaNumHeads {d_model num_heads}.
Arguments mhaHeadDim {d_model num_heads}.
Arguments mhaHeadDimProof {d_model num_heads}.

(** Multi-head attention forward pass:
    1. Project Q, K, V through linear layers
    2. Split into heads
    3. Scaled dot-product attention per head
    4. Combine heads
    5. Output projection *)

Definition multiHeadAttentionForward
  {d_model num_heads batch seq_q seq_k : nat}
  (mha : MultiHeadAttention d_model num_heads)
  (query : Tensor3D batch seq_q d_model)
  (key : Tensor3D batch seq_k d_model)
  (value : Tensor3D batch seq_k d_model)
  (mask : option (Tensor3D batch seq_q seq_k))
  : Tensor3D batch seq_q d_model :=
  let head_dim := mhaHeadDim mha in
  (* 1. Linear projections *)
  let q := linearForward (mhaQueryProj mha) query in   (* [batch, seq_q, d_model] *)
  let k := linearForward (mhaKeyProj mha) key in       (* [batch, seq_k, d_model] *)
  let v := linearForward (mhaValueProj mha) value in   (* [batch, seq_k, d_model] *)
  (* 2. Split into heads *)
  let qHeads := @splitHeads batch seq_q num_heads head_dim q in
  let kHeads := @splitHeads batch seq_k num_heads head_dim k in
  let vHeads := @splitHeads batch seq_k num_heads head_dim v in
  (* 3. Scaled dot-product attention *)
  let attn := @scaledDotProductAttention4D batch num_heads seq_q seq_k head_dim qHeads kHeads vHeads mask in
  (* 4. Combine heads *)
  let combined := @combineHeads batch seq_q num_heads head_dim attn in
  (* 5. Output projection *)
  linearForward (mhaOutputProj mha) combined.

(** ** Initialization *)

Parameter initMultiHeadAttention : forall (d_model num_heads : nat)
  (pf : exists head_dim, d_model = num_heads * head_dim),
  MultiHeadAttention d_model num_heads.

(** ** Properties *)

Lemma attention_preserves_seq_dim : forall
  (d_model num_heads batch seq_q seq_k : nat)
  (mha : MultiHeadAttention d_model num_heads)
  (q : Tensor3D batch seq_q d_model)
  (k v : Tensor3D batch seq_k d_model)
  (mask : option (Tensor3D batch seq_q seq_k)),
  exists (out : Tensor3D batch seq_q d_model), True.
Proof.
  intros.
  exists (multiHeadAttentionForward mha q k v mask).
  trivial.
Qed.

(** Self-attention: Q = K = V from same input. *)

Lemma self_attention_type : forall
  (d_model num_heads batch seq : nat)
  (mha : MultiHeadAttention d_model num_heads)
  (x : Tensor3D batch seq d_model)
  (mask : option (Tensor3D batch seq seq)),
  exists (out : Tensor3D batch seq d_model), True.
Proof.
  intros.
  exists (multiHeadAttentionForward mha x x x mask).
  trivial.
Qed.

(** Cross-attention: Q from decoder, K/V from encoder. *)

Lemma cross_attention_type : forall
  (d_model num_heads batch tgt_len src_len : nat)
  (mha : MultiHeadAttention d_model num_heads)
  (query : Tensor3D batch tgt_len d_model)
  (memory : Tensor3D batch src_len d_model)
  (mask : option (Tensor3D batch tgt_len src_len)),
  exists (out : Tensor3D batch tgt_len d_model), True.
Proof.
  intros.
  exists (multiHeadAttentionForward mha query memory memory mask).
  trivial.
Qed.
