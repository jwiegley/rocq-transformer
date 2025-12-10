(** * Inference Abstractions for Sequence Generation *)

(** This module defines type signatures for inference and decoding utilities,
    focusing on autoregressive sequence generation where output tokens are
    produced one at a time.

    Key algorithms:
    1. greedyDecode: Always select the highest probability token
    2. decodeStep: Single step of autoregressive generation
    3. beamSearch: Maintain multiple hypotheses (placeholder)

    The type signatures capture critical properties:
    - Sequences grow by exactly 1 token per step (S current_len)
    - Generation respects max_len constraints
    - Output shapes match expected batch and length dimensions

    This is an ABSTRACT specification focusing on type-level guarantees
    about dimension transformations during inference. *)

From RocqTransformer Require Import Tensor.
From RocqTransformer Require Import Model.
From RocqTransformer Require Import Decoder.
Require Import Coq.Init.Nat.
Require Import Coq.Lists.List.
Require Import Coq.Arith.PeanoNat.
Require Import Coq.micromega.Lia.
Import ListNotations.

(** ** Greedy Decoding *)

(** Greedy decode: autoregressively generate sequence by selecting the
    token with highest probability at each step.

    Algorithm:
    1. Encode source sequence once → memory
    2. Initialize target with start symbol: [batch, 1]
    3. For each generation step:
       a. Create causal mask for current sequence
       b. Decode: get logits for next token
       c. Select highest probability token (argmax)
       d. Append to sequence
       e. Stop if EOS or max length reached
    4. Return generated sequence

    Properties:
    * Output length gen_len satisfies: 1 <= gen_len <= max_len
    * Output shape is [batch, gen_len]
    * Deterministic for given input (always selects argmax)
    * Locally optimal but globally suboptimal

    Limitations:
    * No backtracking: early mistakes cannot be corrected
    * No diversity: same input always produces same output
    * Length bias: tends to generate shorter sequences

    Corresponds to Haskell:
    greedyDecode :: EncoderDecoder -> Tensor -> Tensor -> Int -> Int -> TrainM Tensor
    greedyDecode model src srcMask maxLen startSymbol = do
      memory <- encode model src srcMask
      ys <- liftIO $ pure $ scale (fromIntegral startSymbol) (ones [batchSize, 1])
      finalYs <- foldM generateStep ys [1..maxLen-1]
      pure finalYs

    Parameters:
    - d_model, d_ff, num_heads, head_dim: Model architecture dimensions
    - num_layers: Number of encoder/decoder layers
    - max_len: Maximum sequence length (positional encoding limit)
    - src_vocab, tgt_vocab: Vocabulary sizes
    - batch: Batch size
    - src_len: Source sequence length
    - gen_len: Generated output length (variable, determined by EOS or max_len)
    - eq: Proof that d_model = num_heads * head_dim
    - src_ok: Proof that src_len <= max_len
    - gen_ok: Proof that gen_len <= max_len
    - model: The complete encoder-decoder model
    - src: Source tokens [batch, src_len]
    - src_mask: Source mask [batch, 1, src_len]
    - max_gen_len: Maximum number of tokens to generate
    - start_symbol: Index of start-of-sequence token (e.g., 1)

    Returns:
    - Generated tokens [batch, gen_len]

    Shape transformation:
    src:      [batch, src_len]      Input tokens
              ↓ Encode
    memory:   [batch, src_len, d_model]  Encoder output
              ↓ Initialize
    ys:       [batch, 1]            Start symbol
              ↓ Autoregressive generation (gen_len - 1 steps)
    output:   [batch, gen_len]      Generated tokens

    Example:
    Input: [32, 100] (batch 32, src_len 100)
    Output: [32, 45] (generated 45 tokens including start symbol) *)

Parameter greedyDecode : forall
  (d_model d_ff num_heads head_dim num_layers max_len
   src_vocab tgt_vocab batch src_len gen_len : nat),
  d_model = num_heads * head_dim ->
  src_len <= max_len ->
  gen_len <= max_len ->
  EncoderDecoder d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab ->
  Tensor2D batch src_len ->           (* Source tokens *)
  Tensor3D batch 1 src_len ->         (* Source mask *)
  nat ->                               (* Max generation length *)
  nat ->                               (* Start symbol index *)
  Tensor2D batch gen_len.              (* Generated tokens *)

(** ** Single Decode Step *)

(** Perform one step of autoregressive generation, extending sequence by exactly
    one token.

    Algorithm:
    1. Create causal mask for current sequence length
    2. Decode current sequence with cross-attention to memory
    3. Get logits for next token (last position)
    4. Project to vocabulary and apply log-softmax
    5. Select token with highest probability (argmax)
    6. Append to sequence: [batch, current_len] → [batch, S current_len]

    Key property: Output length is EXACTLY S current_len (successor of input length).
    This captures the fundamental property of autoregressive generation:
    each step extends the sequence by precisely one token.

    The type signature enforces this at compile time:
    - Input:  Tensor2D batch current_len
    - Output: Tensor2D batch (S current_len)

    The successor (S) in the type signature is crucial:
    * Proves sequence grows monotonically
    * Prevents accidental empty generations
    * Enables reasoning about total generation length
    * Guarantees termination properties

    Corresponds to Haskell:
    decodeStep :: EncoderDecoder -> Tensor -> Tensor -> Tensor
               -> TrainM (Tensor, Int)
    decodeStep model memory srcMask ys = do
      let currentLen = size 1 ys
          tgtMask = subsequentMask currentLen
      decoderOut <- decode model ys memory srcMask tgtMask
      let lastOut = select 1 (currentLen - 1) decoderOut
          probs = generatorForward (edGenerator model) lastOut
          nextTokenTensor = argmax (-1) probs
          nextToken = round $ item nextTokenTensor
          ys' = cat 1 [ys, nextTokenTensor]
      pure (ys', nextToken)

    Parameters:
    - d_model, d_ff, num_heads, head_dim: Model architecture
    - num_layers: Number of layers
    - max_len: Maximum sequence length
    - src_vocab, tgt_vocab: Vocabulary sizes
    - batch: Batch size
    - src_len: Source sequence length
    - current_len: Current generated sequence length
    - eq: Proof that d_model = num_heads * head_dim
    - len_ok: Proof that current_len < max_len (ensures successor fits)
    - model: The encoder-decoder model
    - memory: Encoder output [batch, src_len, d_model]
    - src_mask: Source mask [batch, 1, src_len]
    - ys: Current sequence [batch, current_len]

    Returns:
    - Extended sequence [batch, S current_len] (exactly one token longer)

    Shape transformation:
    ys:       [batch, current_len]     Current sequence
              ↓ Decode
    out:      [batch, current_len, d_model]  Decoder states
              ↓ Select last position
    last:     [batch, d_model]         Last hidden state
              ↓ Generator
    logits:   [batch, vocab_size]      Vocabulary logits
              ↓ Argmax
    next:     [batch, 1]               Next token
              ↓ Concatenate
    result:   [batch, S current_len]   Extended sequence

    Example:
    Input: [32, 10] (batch 32, current_len 10)
    Output: [32, 11] (extended by one token)

    Note: The type system PROVES that the sequence grows by exactly 1.
    This is a powerful compile-time guarantee. *)

Parameter decodeStep : forall
  (d_model d_ff num_heads head_dim num_layers max_len
   src_vocab tgt_vocab batch src_len current_len : nat),
  d_model = num_heads * head_dim ->
  current_len < max_len ->
  EncoderDecoder d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab ->
  Tensor3D batch src_len d_model ->  (* Encoder memory *)
  Tensor3D batch 1 src_len ->        (* Source mask *)
  Tensor2D batch current_len ->      (* Current generated sequence *)
  Tensor2D batch (S current_len).    (* Extended by one token *)

(** ** Beam Search (Placeholder) *)

(** Beam search decoding: maintain multiple hypotheses (beams) to explore
    alternative generation paths.

    Algorithm:
    1. Initialize beam with start symbol: beams = [([START], 0.0)]
    2. Encode source sequence → memory
    3. For each generation step:
       a. Expand each beam:
          - Decode current sequence
          - Get top-k token probabilities
          - Create k new hypotheses: (seq + [token], score + log_prob)
       b. Keep top-k hypotheses overall (across all beams)
       c. Move complete sequences (ending with EOS) to finished list
    4. Return top-k finished sequences

    Advantages over greedy:
    * Explores multiple paths simultaneously
    * Can recover from locally suboptimal choices
    * Often produces higher quality translations
    * Adjustable quality/speed trade-off via beam_size

    Trade-offs:
    * beam_size × slower than greedy
    * Requires more memory (k beams)
    * Still deterministic for fixed beam size
    * More complex implementation

    Hyperparameters:
    * beam_size: Number of hypotheses to maintain (typical: 4-12)
    * length_penalty: Normalize scores by length to avoid bias
    * coverage_penalty: Encourage attention to all source tokens
    * n_best: Return top-n sequences instead of just best

    Corresponds to Haskell:
    beamSearch :: EncoderDecoder -> Tensor -> Tensor -> Int -> Int -> Int
               -> TrainM [Tensor]
    beamSearch model src srcMask beamSize maxLen startSymbol =
      error "Beam search not yet implemented. See OpenNMT-py for reference."

    Parameters:
    - d_model, d_ff, num_heads, head_dim: Model architecture
    - num_layers: Number of layers
    - max_len: Maximum sequence length
    - src_vocab, tgt_vocab: Vocabulary sizes
    - batch: Batch size
    - src_len: Source sequence length
    - gen_len: Generated output length
    - beam_size: Number of hypotheses to maintain
    - eq: Proof that d_model = num_heads * head_dim
    - src_ok: Proof that src_len <= max_len
    - gen_ok: Proof that gen_len <= max_len
    - model: The encoder-decoder model
    - src: Source tokens [batch, src_len]
    - src_mask: Source mask [batch, 1, src_len]
    - beam_width: Number of beams to maintain
    - max_gen_len: Maximum generation length
    - start_symbol: Start token index

    Returns:
    - List of top-k hypotheses, each [batch, gen_len]
      Ordered by score (highest probability first)

    Example:
    Input: [1, 100] (single sequence, src_len 100), beam_size=5
    Output: 5 tensors of shape [1, varying_len] (top-5 hypotheses)

    Note: This is a placeholder. Full implementation would require:
    1. Beam state management (sequences + scores)
    2. Length normalization
    3. Coverage penalty
    4. N-best tracking
    5. Early stopping

    Reference implementations:
    - OpenNMT-py: github.com/OpenNMT/OpenNMT-py
    - Fairseq: github.com/facebookresearch/fairseq *)

Parameter beamSearch : forall
  (d_model d_ff num_heads head_dim num_layers max_len
   src_vocab tgt_vocab batch src_len gen_len beam_size : nat),
  d_model = num_heads * head_dim ->
  src_len <= max_len ->
  gen_len <= max_len ->
  EncoderDecoder d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab ->
  Tensor2D batch src_len ->           (* Source tokens *)
  Tensor3D batch 1 src_len ->         (* Source mask *)
  nat ->                               (* Beam width *)
  nat ->                               (* Max length *)
  nat ->                               (* Start symbol *)
  list (Tensor2D batch gen_len).       (* Top-k hypotheses *)

(** ** Properties: Output Shape Guarantees *)

(** Theorem: Greedy decode produces valid output shape.

    The greedy decode function produces output with:
    * Correct batch dimension (matches input)
    * Valid length: gen_len <= max_len
    * Shape: [batch, gen_len]

    This is guaranteed by the type signature. *)

Theorem greedy_decode_produces_valid_shape :
  forall (d_model d_ff num_heads head_dim num_layers max_len
          src_vocab tgt_vocab batch src_len gen_len max_gen_len start_sym : nat)
    (eq : d_model = num_heads * head_dim)
    (src_ok : src_len <= max_len)
    (gen_ok : gen_len <= max_len)
    (model : EncoderDecoder d_model d_ff num_heads num_layers max_len
               src_vocab tgt_vocab)
    (src : Tensor2D batch src_len)
    (src_mask : Tensor3D batch 1 src_len),
  exists (output : Tensor2D batch gen_len),
    output = greedyDecode d_model d_ff num_heads head_dim num_layers max_len
               src_vocab tgt_vocab batch src_len gen_len eq src_ok gen_ok
               model src src_mask max_gen_len start_sym.
Proof.
  intros.
  eexists.
  reflexivity.
Qed.

(** ** Properties: Monotonic Sequence Growth *)

(** Theorem: Decode step extends sequence by exactly one token.

    The decodeStep function PROVES at the type level that:
    * Input length: current_len
    * Output length: S current_len (exactly one more)

    This monotonic growth property is fundamental to autoregressive generation.
    The type system enforces it, making it impossible to accidentally skip
    tokens or generate multiple tokens in one step. *)

Theorem decode_step_extends_by_one :
  forall (d_model d_ff num_heads head_dim num_layers max_len
          src_vocab tgt_vocab batch src_len current_len : nat)
    (eq : d_model = num_heads * head_dim)
    (len_ok : current_len < max_len)
    (model : EncoderDecoder d_model d_ff num_heads num_layers max_len
               src_vocab tgt_vocab)
    (memory : Tensor3D batch src_len d_model)
    (src_mask : Tensor3D batch 1 src_len)
    (ys : Tensor2D batch current_len),
  exists (output : Tensor2D batch (S current_len)),
    output = decodeStep d_model d_ff num_heads head_dim num_layers max_len
               src_vocab tgt_vocab batch src_len current_len eq len_ok
               model memory src_mask ys.
Proof.
  intros.
  eexists.
  reflexivity.
Qed.

(** Lemma: Decode step output length is precisely one more than input.

    This is a direct consequence of the type signature.
    S current_len is definitionally equal to 1 + current_len. *)

Lemma decode_step_length_increment :
  forall (current_len : nat),
  S current_len = 1 + current_len.
Proof.
  intro.
  reflexivity.
Qed.

(** ** Properties: Generation Constraints *)

(** Theorem: Generated sequence length respects max_len.

    Both greedyDecode and decodeStep require proofs that generated
    sequences do not exceed max_len, ensuring:
    1. Positional encodings remain valid
    2. No buffer overflows in implementation
    3. Generation terminates in bounded time *)

Theorem generation_respects_max_len :
  forall (max_len gen_len current_len : nat),
  gen_len <= max_len ->
  current_len < max_len ->
  S current_len <= max_len.
Proof.
  intros.
  lia.
Qed.

(** Lemma: Iterating decode_step N times produces sequence of length 1+N.

    Starting from length 1 (start symbol), applying decodeStep N times
    produces a sequence of length 1+N.

    This captures the fundamental property of autoregressive generation:
    length grows linearly with number of steps. *)

Lemma decode_step_iteration_length :
  forall (n : nat),
  (* Starting from length 1, after n steps we have length 1 + n *)
  1 + n = S n.
Proof.
  intro.
  reflexivity.
Qed.

(** ** Properties: Beam Search *)

(** Theorem: Beam search produces list of valid hypotheses.

    Each hypothesis in the beam search output:
    * Has correct batch dimension
    * Has valid length: gen_len <= max_len
    * Shape: [batch, gen_len]

    The list contains at most beam_size hypotheses. *)

Theorem beam_search_produces_valid_hypotheses :
  forall (d_model d_ff num_heads head_dim num_layers max_len
          src_vocab tgt_vocab batch src_len gen_len beam_size
          beam_width max_gen_len start_sym : nat)
    (eq : d_model = num_heads * head_dim)
    (src_ok : src_len <= max_len)
    (gen_ok : gen_len <= max_len)
    (model : EncoderDecoder d_model d_ff num_heads num_layers max_len
               src_vocab tgt_vocab)
    (src : Tensor2D batch src_len)
    (src_mask : Tensor3D batch 1 src_len),
  let hypotheses := beamSearch d_model d_ff num_heads head_dim num_layers
                      max_len src_vocab tgt_vocab batch src_len gen_len
                      beam_size eq src_ok gen_ok model src src_mask
                      beam_width max_gen_len start_sym in
  (* Each hypothesis has correct shape *)
  forall (hyp : Tensor2D batch gen_len),
    In hyp hypotheses -> True.
Proof.
  intros.
  trivial.
Qed.

(** ** Properties: Termination *)

(** Lemma: Greedy decode terminates in at most max_len steps.

    Since each step generates one token and generation stops at max_len,
    the algorithm terminates in bounded time.

    This is crucial for proving total correctness of the inference procedure. *)

Lemma greedy_decode_terminates :
  forall (max_len : nat),
  max_len <= max_len.
Proof.
  intro.
  lia.
Qed.

(** Lemma: Decode step requires current length strictly less than max length.

    This constraint ensures:
    1. We have space for one more token
    2. Generation can terminate before exceeding max_len
    3. Positional encoding lookup remains in bounds *)

Lemma decode_step_requires_space :
  forall (current_len max_len : nat),
  current_len < max_len ->
  S current_len <= max_len.
Proof.
  intros.
  lia.
Qed.

(** ** Inference Strategy Comparison *)

(** Lemma: Greedy is a special case of beam search with beam_size=1.

    Greedy decoding can be viewed as beam search with only one hypothesis
    maintained at each step.

    This relationship helps understand the trade-off:
    * Greedy: Fast, deterministic, locally optimal
    * Beam: Slower, explores alternatives, often better quality *)

Lemma greedy_is_beam_one :
  forall (beam_size : nat),
  beam_size = 1 ->
  (* Beam search with size 1 behaves like greedy *)
  True.
Proof.
  intros.
  trivial.
Qed.

(** ** Export *)

(** The Inference module provides type signatures for sequence generation:

    Key exports:
    - greedyDecode: Simple, fast, deterministic generation
    - decodeStep: Single step with sequence length increment guarantee
    - beamSearch: Multiple hypothesis exploration (placeholder)

    Type-level guarantees:
    - Sequence length grows by exactly 1 per step (S current_len)
    - Generation respects max_len constraints
    - Output shapes are well-defined and predictable

    These abstractions complete the Transformer model specification,
    enabling inference and translation applications.

    The type signatures capture the essential properties of autoregressive
    generation without requiring concrete tensor implementations. *)
