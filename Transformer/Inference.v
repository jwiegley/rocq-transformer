(** * Inference: Autoregressive Sequence Generation *)

(** Greedy decoding generates output tokens one at a time:
    1. Encode source once
    2. Initialize with start token
    3. Loop: decode -> argmax -> append until max_len *)

From Transformer Require Import Tensor.
From Transformer Require Import Model.
From Transformer Require Import Decoder.
From Transformer Require Import Embedding.
From Transformer Require Import Linear.
From Stdlib Require Import Init.Nat.
From Stdlib Require Import Lists.List.
From Stdlib Require Import Arith.PeanoNat.
From Stdlib Require Import Lia.
Import ListNotations.

(** ** Arithmetic Lemmas for Type Coercions *)

(** These lemmas help with the type-level arithmetic needed when
    the sequence length changes during generation. *)

Lemma add_0_r_eq : forall n, n + 0 = n.
Proof. apply Nat.add_0_r. Qed.

Lemma add_succ_comm : forall n m, S n + m = n + S m.
Proof. intros. lia. Qed.

Lemma add_1_r_eq : forall n, n + 1 = S n.
Proof. intros. lia. Qed.

(** Type-safe transport for tensors along dimension equality. *)

Definition transport2D {batch d1 d2 : nat} (eq : d1 = d2)
  (t : Tensor2D batch d1) : Tensor2D batch d2 :=
  match eq in (_ = d) return Tensor2D batch d with
  | eq_refl => t
  end.

(** ** Single Decode Step *)

(** One step of autoregressive generation.
    Given current output of length cur_len, produce the next token [batch, 1]. *)

Definition decodeStepSimple
  {d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab batch src_len cur_len : nat}
  (model : EncoderDecoder d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab)
  (memory : Tensor3D batch src_len d_model)
  (tgtSoFar : Tensor2D batch cur_len)
  (curLenProof : cur_len <= max_len)
  : Tensor2D batch 1 :=
  (* Create causal mask for current sequence length *)
  let causalMask := subsequentMask cur_len in
  let tgtMask := expand_mask2D_to_3D batch cur_len causalMask in
  (* Embed target tokens *)
  let embedded := embeddingsForward (edTgtEmbed model) tgtSoFar in
  let withPos := positionalEncodingForward (edTgtPos model) embedded curLenProof in
  (* Decode with cross-attention to encoder memory *)
  let decoded := decoderForward (edDecoder model) withPos memory (Some tgtMask) None in
  (* Get logits via generator: [batch, cur_len, tgt_vocab] *)
  let logits := generatorForward (edGenerator model) decoded in
  (* Extract last position and argmax to get next token: [batch, 1] *)
  argmax_last_position batch cur_len tgt_vocab logits.

(** ** Greedy Decode Loop *)

(** The core iteration of greedy decoding.
    Given the current sequence of length curLen and number of tokens remaining
    to generate, produces a sequence of length curLen + remaining.

    This uses structural recursion on 'remaining' with type coercions
    to handle the arithmetic at the type level. *)

(** Helper to construct proof that S curLen <= max_len from curLen + S rem' <= max_len *)
Definition derive_new_len_proof (curLen rem' max_len : nat)
  (pf : curLen + S rem' <= max_len) : S curLen <= max_len.
Proof.
  assert (H: S curLen <= curLen + S rem') by lia.
  exact (Nat.le_trans _ _ _ H pf).
Defined.

(** Helper to construct proof that S curLen + rem' <= max_len from curLen + S rem' <= max_len *)
Definition derive_rem_proof (curLen rem' max_len : nat)
  (pf : curLen + S rem' <= max_len) : S curLen + rem' <= max_len.
Proof.
  assert (H: S curLen + rem' = curLen + S rem') by lia.
  rewrite H. exact pf.
Defined.

Fixpoint greedyDecodeLoop
  {d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab batch src_len : nat}
  (model : EncoderDecoder d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab)
  (memory : Tensor3D batch src_len d_model)
  (remaining : nat)
  (curLen : nat)
  (tgtSoFar : Tensor2D batch curLen)
  (curLenPf : curLen <= max_len)
  (totalPf : curLen + remaining <= max_len)
  {struct remaining}
  : Tensor2D batch (curLen + remaining) :=
  match remaining as r return (curLen + r <= max_len) -> Tensor2D batch (curLen + r) with
  | 0 => fun _ =>
      (* Base case: no more tokens to generate *)
      (* Need to coerce curLen + 0 = curLen *)
      transport2D (eq_sym (add_0_r_eq curLen)) tgtSoFar
  | S rem' => fun pf =>
      (* Generate one token *)
      let nextToken := decodeStepSimple model memory tgtSoFar curLenPf in
      (* Concatenate: [batch, curLen] ++ [batch, 1] -> [batch, curLen + 1] *)
      let tgtNewRaw := cat2D_seq batch curLen 1 tgtSoFar nextToken in
      (* Coerce curLen + 1 to S curLen *)
      let tgtNew := transport2D (add_1_r_eq curLen) tgtNewRaw in
      (* Recurse with one fewer remaining token *)
      let newPf := derive_new_len_proof curLen rem' max_len pf in
      let remPf := derive_rem_proof curLen rem' max_len pf in
      let result := greedyDecodeLoop model memory rem' (S curLen) tgtNew newPf remPf in
      (* Coerce type: (S curLen) + rem' to curLen + (S rem') *)
      (* add_succ_comm curLen rem' : S curLen + rem' = curLen + S rem' *)
      transport2D (add_succ_comm curLen rem') result
  end totalPf.

(** ** Greedy Decode *)

(** Complete greedy decoding from source tokens to generated output.

    Algorithm:
    1. Encode source sequence to get memory [batch, src_len, d_model]
    2. Initialize with start token [batch, 1]
    3. Iterate gen_len - 1 times:
       - Create causal mask for current sequence
       - Decode: embed -> positional -> decoder -> generator
       - Argmax to get next token
       - Append to sequence
    4. Return [batch, gen_len] generated tokens *)

(** Helper to derive that 1 <= max_len from gen_len >= 1 and gen_len <= max_len *)
Definition derive_one_le_max (gen_len max_len : nat)
  (genPos : gen_len >= 1) (genPf : gen_len <= max_len) : 1 <= max_len.
Proof.
  exact (Nat.le_trans _ _ _ genPos genPf).
Defined.

(** Helper to derive that 1 + (gen_len - 1) <= max_len from gen_len <= max_len and gen_len >= 1 *)
Definition derive_total_proof (gen_len max_len : nat)
  (genPos : gen_len >= 1) (genPf : gen_len <= max_len) : 1 + (gen_len - 1) <= max_len.
Proof.
  assert (H: 1 + (gen_len - 1) = gen_len) by lia.
  rewrite H. exact genPf.
Defined.

(** Helper lemma: 1 + (n - 1) = n when n >= 1 *)
Lemma one_plus_pred : forall n, n >= 1 -> 1 + (n - 1) = n.
Proof. intros. lia. Qed.

Definition greedyDecode
  {d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab batch src_len gen_len : nat}
  (model : EncoderDecoder d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab)
  (src : Tensor2D batch src_len)
  (srcMask : option (Tensor3D batch src_len src_len))
  (startToken : nat)
  (srcLenProof : src_len <= max_len)
  (genLenProof : gen_len <= max_len)
  (genLenPos : gen_len >= 1)
  : Tensor2D batch gen_len :=
  (* Step 1: Encode source sequence *)
  let memory := encode model src srcMask srcLenProof in
  (* Step 2: Initialize with start token [batch, 1] *)
  let startSeq := fill2D batch 1 startToken in
  (* Step 3: Generate remaining tokens via the loop *)
  (* We need gen_len total, starting with 1, so generate gen_len - 1 more *)
  let remaining := gen_len - 1 in
  let onePf := derive_one_le_max gen_len max_len genLenPos genLenProof in
  let totalPf := derive_total_proof gen_len max_len genLenPos genLenProof in
  let result := greedyDecodeLoop model memory remaining 1 startSeq onePf totalPf in
  (* Transport result type: 1 + (gen_len - 1) = gen_len *)
  transport2D (one_plus_pred gen_len genLenPos) result.

(** ** Alternative: Direct decodeStep with mask parameter *)

(** One step of autoregressive generation with explicit mask.
    Used when more control over masking is needed. *)

Definition decodeStep
  {d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab batch src_len cur_len : nat}
  (model : EncoderDecoder d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab)
  (memory : Tensor3D batch src_len d_model)
  (tgtSoFar : Tensor2D batch cur_len)
  (tgtMask : Tensor3D batch cur_len cur_len)
  (srcMask : option (Tensor3D batch cur_len src_len))
  (curLenProof : cur_len <= max_len)
  : Tensor2D batch 1 :=
  (* Embed target so far *)
  let embedded := embeddingsForward (edTgtEmbed model) tgtSoFar in
  let withPos := positionalEncodingForward (edTgtPos model) embedded curLenProof in
  (* Decode *)
  let decoded := decoderForward (edDecoder model) withPos memory (Some tgtMask) srcMask in
  (* Get logits via generator *)
  let logits := generatorForward (edGenerator model) decoded in
  (* Extract last position, argmax *)
  argmax_last_position batch cur_len tgt_vocab logits.

(** ** Properties *)

Lemma greedy_decode_output_shape : forall
  (d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab batch src_len gen_len : nat)
  (model : EncoderDecoder d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab)
  (src : Tensor2D batch src_len)
  (srcMask : option (Tensor3D batch src_len src_len))
  (startToken : nat)
  (srcPf : src_len <= max_len)
  (genPf : gen_len <= max_len)
  (genPos : gen_len >= 1),
  exists (y : Tensor2D batch gen_len), True.
Proof.
  intros.
  exists (greedyDecode model src srcMask startToken srcPf genPf genPos).
  trivial.
Qed.

Lemma decode_step_produces_one_token : forall
  (d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab batch src_len cur_len : nat)
  (model : EncoderDecoder d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab)
  (memory : Tensor3D batch src_len d_model)
  (tgtSoFar : Tensor2D batch cur_len)
  (tgtMask : Tensor3D batch cur_len cur_len)
  (srcMask : option (Tensor3D batch cur_len src_len))
  (curPf : cur_len <= max_len),
  exists (nextToken : Tensor2D batch 1), True.
Proof.
  intros.
  exists (decodeStep model memory tgtSoFar tgtMask srcMask curPf).
  trivial.
Qed.

Lemma greedy_loop_length : forall
  (d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab batch src_len : nat)
  (model : EncoderDecoder d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab)
  (memory : Tensor3D batch src_len d_model)
  (remaining curLen : nat)
  (tgt : Tensor2D batch curLen)
  (curPf : curLen <= max_len)
  (totalPf : curLen + remaining <= max_len),
  exists (result : Tensor2D batch (curLen + remaining)), True.
Proof.
  intros.
  exists (greedyDecodeLoop model memory remaining curLen tgt curPf totalPf).
  trivial.
Qed.
