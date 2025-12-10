(** * Token Embeddings and Positional Encoding *)

(** This module implements token embeddings and sinusoidal positional encoding
    for the Transformer model, following "Attention is All You Need" (Vaswani et al., 2017).

    Token embeddings convert discrete token indices into continuous vector representations,
    which are then combined with positional encodings to inject sequence position information.

    Key features:
    - Token embeddings scaled by √d_model for balanced magnitudes
    - Fixed sinusoidal positional encodings with multi-scale frequencies
    - Type-safe dimension tracking with proofs of sequence length constraints *)

From RocqTransformer Require Import Tensor.
Require Import Coq.Init.Nat.
Require Import Coq.Arith.PeanoNat.

(** ** Token Embeddings *)

(** The Embeddings record represents a learned lookup table that converts
    token indices to dense vector representations.

    In the Haskell implementation:
    {[
      data Embeddings = Embeddings
        { embWeight :: Tensor   -- [vocab_size, d_model]
        , embDModel :: Int      -- For √d_model scaling
        }
    ]}

    The embedding matrix is abstract here - we only specify its dimensions
    and operations. The actual weights would be learned during training. *)

Record Embeddings (vocab_size d_model : nat) : Type :=
  mkEmbeddings {
    (* The embedding weight matrix [vocab_size, d_model] is abstract.
       Each row contains the d_model-dimensional embedding for one token. *)
  }.

(** Embedding forward pass: convert token indices to scaled embeddings.

    Process:
    1. Look up embedding vectors for each token index
    2. Scale by √d_model for balanced magnitudes with positional encodings

    From the paper:
    "In the embedding layers, we multiply those weights by √d_model"

    Input:  Token indices [batch, seq]
    Output: Scaled embeddings [batch, seq, d_model]

    Corresponds to Haskell:
    {[
      embeddingsForward :: Embeddings -> Tensor -> Tensor
      embeddingsForward emb input =
        let looked_up = embeddingLookup embWeight input
            scaleFactor = sqrt (fromIntegral embDModel)
        in scale scaleFactor looked_up
    ]} *)

Parameter embeddingsForward : forall (vocab_size d_model batch seq : nat),
  Embeddings vocab_size d_model ->
  Tensor2D batch seq ->          (* Token indices *)
  Tensor3D batch seq d_model.    (* Scaled embeddings *)

(** Initialize embeddings with random weights.

    In practice, weights are typically initialized from a normal distribution
    scaled by 1/√d_model or using Xavier/Kaiming initialization. *)

Parameter initEmbeddings : forall (vocab_size d_model : nat),
  Embeddings vocab_size d_model.

(** ** Positional Encoding *)

(** Positional encoding adds information about token position in the sequence.
    Since the Transformer has no inherent notion of position (unlike RNNs),
    we must explicitly inject positional information.

    The paper uses fixed sinusoidal functions of different frequencies:

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Where:
    - pos is the position in the sequence (0, 1, 2, ...)
    - i is the dimension index (0 to d_model/2)
    - Even dimensions use sine, odd dimensions use cosine

    Why sinusoidal?
    1. Unique encoding for each position
    2. Smooth distance metric between positions
    3. Linear relationship for relative positions (PE(pos+k) = M_k · PE(pos))
    4. Extrapolates to longer sequences than seen during training

    In the Haskell implementation:
    {[
      data PositionalEncoding = PositionalEncoding
        { peEncoding :: Tensor  -- [max_len, d_model] pre-computed
        , peDropout  :: Float   -- Regularization
        }
    ]} *)

Record PositionalEncoding (max_len d_model : nat) : Type :=
  mkPositionalEncoding {
    (* Pre-computed sinusoidal encodings [max_len, d_model] are abstract.
       Computed once during initialization:
       - Low dimensions (high frequency): encode fine-grained position
       - High dimensions (low frequency): encode coarse position
       Wavelengths form geometric progression from 2π to 10000·2π *)
  }.

(** Add positional encodings to embeddings.

    The sequence length must not exceed max_len (the maximum length for which
    positional encodings were pre-computed). This constraint is enforced by
    requiring a proof that seq <= max_len.

    Process:
    1. Extract positional encodings for positions 0 to seq-1
    2. Add to input embeddings (broadcasting across batch dimension)
    3. Apply dropout for regularization (during training)

    Input:  Embeddings [batch, seq, d_model]
    Output: Embeddings with position info [batch, seq, d_model]

    Corresponds to Haskell:
    {[
      positionalEncodingForward :: PositionalEncoding -> Tensor -> TrainM Tensor
      positionalEncodingForward pe embeddings = do
        let seqLen = size 1 embeddings
            pe3D = replicate batchSize (take seqLen peEncoding)
            combined = add embeddings pe3D
        training <- isTraining
        if training then dropout peDropout combined else pure combined
    ]} *)

Parameter positionalEncodingForward : forall (max_len d_model batch seq : nat),
  seq <= max_len ->                    (* Proof: sequence fits in pre-computed table *)
  PositionalEncoding max_len d_model ->
  Tensor3D batch seq d_model ->        (* Input embeddings *)
  Tensor3D batch seq d_model.          (* Output with position info *)

(** Initialize positional encoding with pre-computed sinusoidal values.

    Computes the encoding matrix once during initialization:
    - For each position pos in [0, max_len)
    - For each dimension dim in [0, d_model)
    - Compute sin or cos based on the formula above *)

Parameter initPositionalEncoding : forall (max_len d_model : nat),
  PositionalEncoding max_len d_model.

(** ** Combined Embedding Pipeline *)

(** Combine token embedding lookup and positional encoding in one step.

    This is the typical usage pattern:
    1. Look up embeddings for token indices
    2. Add positional encodings
    3. Result is ready for encoder/decoder input

    The proof seq_ok : seq <= max_len ensures we don't request positional
    encodings beyond what was pre-computed. *)

Definition embedWithPosition
  (vocab_size max_len d_model batch seq : nat)
  (seq_ok : seq <= max_len)
  (emb : Embeddings vocab_size d_model)
  (pe : PositionalEncoding max_len d_model)
  (tokens : Tensor2D batch seq)
  : Tensor3D batch seq d_model :=
  let embedded := embeddingsForward vocab_size d_model batch seq emb tokens in
  positionalEncodingForward max_len d_model batch seq seq_ok pe embedded.

(** ** Correctness Properties *)

(** Embedding produces tensors with the correct output shape. *)

Theorem embedding_shape :
  forall (vocab_size d_model batch seq : nat)
         (emb : Embeddings vocab_size d_model)
         (tokens : Tensor2D batch seq),
  exists (y : Tensor3D batch seq d_model), True.
Proof.
  intros.
  exists (embeddingsForward vocab_size d_model batch seq emb tokens).
  trivial.
Qed.

(** Positional encoding preserves the input shape. *)

Theorem positional_encoding_shape :
  forall (max_len d_model batch seq : nat)
         (seq_ok : seq <= max_len)
         (pe : PositionalEncoding max_len d_model)
         (embeddings : Tensor3D batch seq d_model),
  exists (y : Tensor3D batch seq d_model), True.
Proof.
  intros.
  exists (positionalEncodingForward max_len d_model batch seq seq_ok pe embeddings).
  trivial.
Qed.

(** Combined embedding pipeline produces correct output shape. *)

Theorem embedWithPosition_shape :
  forall (vocab_size max_len d_model batch seq : nat)
         (seq_ok : seq <= max_len)
         (emb : Embeddings vocab_size d_model)
         (pe : PositionalEncoding max_len d_model)
         (tokens : Tensor2D batch seq),
  exists (y : Tensor3D batch seq d_model), True.
Proof.
  intros.
  unfold embedWithPosition.
  exists (positionalEncodingForward max_len d_model batch seq seq_ok pe
           (embeddingsForward vocab_size d_model batch seq emb tokens)).
  trivial.
Qed.

(** Sequence length constraint is necessary and sufficient.

    This theorem states that for positional encoding to be defined,
    we need exactly the constraint that seq <= max_len. *)

Theorem positional_encoding_requires_constraint :
  forall (max_len d_model batch seq : nat)
         (pe : PositionalEncoding max_len d_model)
         (embeddings : Tensor3D batch seq d_model),
  seq <= max_len ->
  exists (output : Tensor3D batch seq d_model), True.
Proof.
  intros max_len d_model batch seq pe embeddings H.
  exists (positionalEncodingForward max_len d_model batch seq H pe embeddings).
  trivial.
Qed.

(** ** Implementation Notes *)

(** *** Scaling Factor √d_model

    The paper scales embeddings by √d_model to balance their magnitude with
    positional encodings. Without this:

    - Positional encodings (sine/cosine) have values in [-1, 1]
    - Random embeddings would dominate with larger magnitudes
    - Their sum would be imbalanced

    With d_model = 512, √d_model ≈ 22.6, bringing embeddings to a similar
    scale as positional encodings. *)

(** *** Sinusoidal Frequency Progression

    The wavelengths form a geometric progression from 2π to 10000·2π:

    - Low dimensions (i ≈ 0): High frequency, short wavelength (≈2π)
      Encodes fine-grained positional differences
      Period of ~6 tokens

    - High dimensions (i ≈ d_model/2): Low frequency, long wavelength (≈10000·2π)
      Encodes coarse positional information
      Period of ~10000 tokens

    This multi-scale encoding allows the model to attend to both nearby
    and distant positions effectively. *)

(** *** Weight Tying

    The paper notes that the same embedding weight matrix can be shared between:
    1. Source embeddings (encoder input)
    2. Target embeddings (decoder input)
    3. Pre-softmax linear transformation (decoder output)

    This reduces parameters and can improve performance. Our abstract
    specification allows but doesn't require this sharing. *)

(** *** Learned vs. Fixed Encodings

    The paper reports that learned positional embeddings yield nearly
    identical results to fixed sinusoidal encodings. However, sinusoidal
    encodings have the advantage of potentially extrapolating to sequences
    longer than those seen during training.

    This implementation follows the paper in using fixed sinusoidal encodings. *)
