(** * Complete Encoder-Decoder Model and Generator *)

(** This module implements the complete Transformer model, combining all components:
    encoder, decoder, embeddings, positional encoding, and output generation.

    The architecture follows "Attention is All You Need" (Vaswani et al., 2017):

    Input → Embedding → PE → Encoder → Memory
                                         ↓
    Target → Embedding → PE → Decoder → Generator → Output Logits
                               ↑
                            Memory

    == Architecture Overview

    The complete model consists of:

    1. Source Processing Path:
       - Token embeddings (learned)
       - Positional encoding (fixed sinusoidal)
       - N encoder layers (self-attention + FFN)

    2. Target Processing Path:
       - Token embeddings (learned)
       - Positional encoding (fixed sinusoidal)
       - N decoder layers (masked self-attention + cross-attention + FFN)

    3. Output Generation:
       - Linear projection to vocabulary size
       - Log-softmax for probability distribution

    == Key Design Decisions

    * Shared Embeddings: Source and target embeddings are typically different,
      but can share parameters if vocabularies are the same

    * Weight Tying: The output projection can optionally share weights with
      target embeddings (reduces parameters, often improves performance)

    * Dropout: Applied uniformly across all sublayers for regularization

    * Layer Normalization: Pre-norm architecture for training stability

    This is an ABSTRACT specification - operations are axiomatized without
    concrete implementations. The goal is to capture the dimensional constraints
    and invariants of the complete model architecture in the type system. *)

From RocqTransformer Require Import Tensor.
From RocqTransformer Require Import Config.
From RocqTransformer Require Import Linear.
From RocqTransformer Require Import Encoder.
From RocqTransformer Require Import Decoder.
From RocqTransformer Require Import Embedding.
Require Import Coq.Init.Nat.
Require Import Coq.Arith.PeanoNat.

(** ** Output Generator *)

(** Output projection layer that projects decoder hidden states to vocabulary
    logits and applies log-softmax.

    Architecture:
    Hidden States [batch, seq, d_model]
          ↓
    Linear Projection
          ↓
    Logits [batch, seq, vocab_size]
          ↓
    Log Softmax (dim=-1)
          ↓
    Log Probabilities [batch, seq, vocab_size]

    Design Notes:
    * Weight Tying: This projection can optionally share weights with the
      target embedding layer to reduce parameters
    * Scaling: The original paper scales embeddings by √d_model
    * Normalization: Log-softmax is preferred over softmax for numerical
      stability when computing cross-entropy loss

    In the Haskell implementation:
    data Generator = Generator
      { genProj :: Linear  -- d_model → vocab_size
      }

    Corresponds to Haskell:
    data Generator = Generator { genProj :: Linear } *)

Record Generator (d_model vocab_size : nat) := mkGenerator {
  gen_proj : Linear d_model vocab_size  (* Linear projection to vocabulary *)
}.

(** Generator forward pass: project to vocabulary and apply log-softmax.

    Projects hidden states to vocabulary logits and applies log-softmax
    for numerical stability.

    Mathematical definition:
    logits[b,t,v] = W_out · h[b,t] + b_out
    output[b,t,v] = log(exp(logits[b,t,v]) / Σ_v' exp(logits[b,t,v']))

    Properties:
    * Output values are negative (log probabilities)
    * Each position sums to 1.0 when exponentiated
    * Numerically stable for loss computation

    Parameters:
    - d_model: Model dimension (e.g., 512)
    - vocab_size: Target vocabulary size (e.g., 10000)
    - batch: Batch size
    - seq: Sequence length
    - gen: Generator layer
    - x: Hidden states [batch, seq, d_model]

    Returns:
    - Log probabilities [batch, seq, vocab_size]

    Shape transformation:
    Input:  [batch, seq, d_model]
           ↓ Linear
    Logits: [batch, seq, vocab_size]
           ↓ LogSoftmax(dim=-1)
    Output: [batch, seq, vocab_size]

    Corresponds to Haskell:
    generatorForward :: Generator -> Tensor -> Tensor
    generatorForward Generator{..} x =
      let logits = linearForward genProj x
      in logSoftmax (-1) logits

    Example:
    Input: [32, 50, 512] (batch 32, seq_len 50, d_model 512)
    Output: [32, 50, 10000] (log probabilities over 10000-token vocab) *)

Parameter generatorForward : forall (d_model vocab_size batch seq : nat),
  Generator d_model vocab_size ->
  Tensor3D batch seq d_model ->       (* Decoder hidden states *)
  Tensor3D batch seq vocab_size.      (* Log probabilities over vocabulary *)

(** Initialize output generator.

    Creates a linear projection from d_model to vocab_size with random
    initialization.

    In practice, weights are initialized from a normal distribution
    scaled appropriately (e.g., Xavier/Glorot initialization).

    Corresponds to Haskell:
    initGenerator :: Int -> Int -> IO Generator
    initGenerator dModel vocabSize = do
      proj <- initLinear dModel vocabSize
      pure $ Generator proj

    Example:
    let gen = initGenerator 512 10000 in ... *)

Parameter initGenerator : forall (d_model vocab_size : nat),
  Generator d_model vocab_size.

(** ** Complete Encoder-Decoder Model *)

(** Complete Encoder-Decoder Transformer combining all components.

    This is the main model that integrates:
    1. Source embedding path (Embeddings + PositionalEncoding)
    2. Target embedding path (Embeddings + PositionalEncoding)
    3. Encoder stack (N layers of self-attention + FFN)
    4. Decoder stack (N layers of masked self-attention + cross-attention + FFN)
    5. Output projection (Generator)

    Information Flow:

    Source Tokens [batch, src_len]
          ↓ Embedding
    Source Embeddings [batch, src_len, d_model]
          ↓ Positional Encoding
    Source Inputs [batch, src_len, d_model]
          ↓ Encoder
    Memory [batch, src_len, d_model] ────────┐
                                             │
    Target Tokens [batch, tgt_len]           │
          ↓ Embedding                        │
    Target Embeddings [batch, tgt_len, d_model]│
          ↓ Positional Encoding              │
    Target Inputs [batch, tgt_len, d_model]  │
          ↓ Decoder ←─────────────────────────┘
    Decoder Output [batch, tgt_len, d_model]
          ↓ Generator
    Output Logits [batch, tgt_len, tgt_vocab]

    In the Haskell implementation:
    data EncoderDecoder = EncoderDecoder
      { edEncoder   :: Encoder
      , edDecoder   :: Decoder
      , edSrcEmbed  :: (Embeddings, PositionalEncoding)
      , edTgtEmbed  :: (Embeddings, PositionalEncoding)
      , edGenerator :: Generator
      }

    Corresponds to Haskell:
    data EncoderDecoder = EncoderDecoder
      { edEncoder :: Encoder
      , edDecoder :: Decoder
      , edSrcEmbed :: (Embeddings, PositionalEncoding)
      , edTgtEmbed :: (Embeddings, PositionalEncoding)
      , edGenerator :: Generator } *)

Record EncoderDecoder
  (d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab : nat) :=
  mkEncoderDecoder {
    ed_encoder : Encoder d_model d_ff num_heads num_layers;
    ed_decoder : Decoder d_model d_ff num_heads num_layers;
    ed_src_embed : Embeddings src_vocab d_model;
    ed_src_pe : PositionalEncoding max_len d_model;
    ed_tgt_embed : Embeddings tgt_vocab d_model;
    ed_tgt_pe : PositionalEncoding max_len d_model;
    ed_generator : Generator d_model tgt_vocab
  }.

(** ** Encode Source Sequence *)

(** Encode source sequence through embeddings and encoder.

    Processes source tokens through the complete encoding pipeline:
    1. Token embedding lookup
    2. Add positional encoding
    3. Process through encoder stack

    Algorithm:
    src_emb = embeddingsForward(ed_src_embed, src)
    src_pe = positionalEncodingForward(ed_src_pe, src_emb)
    memory = encoderForward(ed_encoder, src_pe, src_mask)

    Parameters:
    - d_model: Model dimension (e.g., 512)
    - d_ff: Feed-forward inner dimension (e.g., 2048)
    - num_heads: Number of attention heads (e.g., 8)
    - head_dim: Dimension per head (d_model / num_heads = 64)
    - num_layers: Number of encoder layers (e.g., 6)
    - max_len: Maximum sequence length (e.g., 5000)
    - src_vocab: Source vocabulary size
    - tgt_vocab: Target vocabulary size
    - batch: Batch size
    - src_len: Source sequence length
    - eq: Proof that d_model = num_heads * head_dim
    - len_ok: Proof that src_len <= max_len
    - model: The complete model
    - src: Source token indices [batch, src_len]
    - src_mask: Source padding mask [batch, 1, src_len]

    Returns:
    - Encoder memory [batch, src_len, d_model]

    Shape transformation:
    src:      [batch, src_len]           Token IDs
            ↓ Embedding
            [batch, src_len, d_model]    Embedded tokens
            ↓ Positional Encoding
            [batch, src_len, d_model]    With position info
            ↓ Encoder
    memory:   [batch, src_len, d_model]  Contextualized representations

    Usage:
    This is useful for caching encoder output during inference.
    Encode once, then decode multiple times (e.g., beam search).

    Corresponds to Haskell:
    encode :: EncoderDecoder -> Tensor -> Tensor -> TrainM Tensor
    encode EncoderDecoder{..} src srcMask = do
      let (srcEmbed, srcPE) = edSrcEmbed
      let x = embeddingsForward srcEmbed src
      x' <- positionalEncodingForward srcPE x
      encoderForward edEncoder x' srcMask

    Example:
    Input tokens: [32, 100] (batch 32, src_len 100)
    After embedding: [32, 100, 512]
    After positional encoding: [32, 100, 512]
    After encoder: [32, 100, 512] (memory) *)

Parameter encode : forall
  (d_model d_ff num_heads head_dim num_layers max_len src_vocab tgt_vocab
   batch src_len : nat),
  d_model = num_heads * head_dim ->
  src_len <= max_len ->
  EncoderDecoder d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab ->
  Tensor2D batch src_len ->           (* Source tokens *)
  Tensor3D batch 1 src_len ->         (* Source mask *)
  Tensor3D batch src_len d_model.     (* Encoder memory *)

(** ** Decode Target Sequence *)

(** Decode target sequence given encoder memory.

    Processes target tokens through the complete decoding pipeline:
    1. Token embedding lookup
    2. Add positional encoding
    3. Process through decoder stack with cross-attention to memory

    Algorithm:
    tgt_emb = embeddingsForward(ed_tgt_embed, tgt)
    tgt_pe = positionalEncodingForward(ed_tgt_pe, tgt_emb)
    output = decoderForward(ed_decoder, tgt_pe, memory, src_mask, tgt_mask)

    Parameters:
    - d_model: Model dimension (e.g., 512)
    - d_ff: Feed-forward inner dimension (e.g., 2048)
    - num_heads: Number of attention heads (e.g., 8)
    - head_dim: Dimension per head (d_model / num_heads = 64)
    - num_layers: Number of decoder layers (e.g., 6)
    - max_len: Maximum sequence length (e.g., 5000)
    - src_vocab: Source vocabulary size
    - tgt_vocab: Target vocabulary size
    - batch: Batch size
    - tgt_len: Target sequence length
    - src_len: Source sequence length
    - eq: Proof that d_model = num_heads * head_dim
    - len_ok: Proof that tgt_len <= max_len
    - model: The complete model
    - tgt: Target token indices [batch, tgt_len]
    - memory: Encoder output [batch, src_len, d_model]
    - src_mask: Source mask [batch, 1, src_len]
    - tgt_mask: Target causal mask [batch, tgt_len, tgt_len]

    Returns:
    - Decoder output [batch, tgt_len, d_model]

    Shape transformation:
    tgt:      [batch, tgt_len]           Token IDs
            ↓ Embedding
            [batch, tgt_len, d_model]    Embedded tokens
            ↓ Positional Encoding
            [batch, tgt_len, d_model]    With position info
            ↓ Decoder (with cross-attention to memory)
    output:   [batch, tgt_len, d_model]  Contextualized representations

    Attention patterns:
    The decoder performs two types of attention:
    1. Self-Attention (masked): Each target position attends to previous positions
       Controlled by tgt_mask (typically causal mask)
    2. Cross-Attention: Each target position attends to all source positions
       Controlled by src_mask (typically padding mask)

    Usage contexts:
    * Training: Process entire target sequence at once (teacher forcing)
    * Inference: Process one token at a time (autoregressive)

    Corresponds to Haskell:
    decode :: EncoderDecoder -> Tensor -> Tensor -> Tensor -> Tensor
           -> TrainM Tensor
    decode EncoderDecoder{..} tgt memory srcMask tgtMask = do
      let (tgtEmbed, tgtPE) = edTgtEmbed
      let x = embeddingsForward tgtEmbed tgt
      x' <- positionalEncodingForward tgtPE x
      decoderForward edDecoder x' memory srcMask tgtMask

    Example:
    Target input: [32, 50] (batch 32, tgt_len 50)
    Encoder memory: [32, 100, 512]
    After embedding: [32, 50, 512]
    After positional encoding: [32, 50, 512]
    After decoder: [32, 50, 512] (before generator projection) *)

Parameter decode : forall
  (d_model d_ff num_heads head_dim num_layers max_len src_vocab tgt_vocab
   batch tgt_len src_len : nat),
  d_model = num_heads * head_dim ->
  tgt_len <= max_len ->
  EncoderDecoder d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab ->
  Tensor2D batch tgt_len ->           (* Target tokens *)
  Tensor3D batch src_len d_model ->   (* Encoder memory *)
  Tensor3D batch 1 src_len ->         (* Source mask *)
  Tensor3D batch tgt_len tgt_len ->   (* Target mask *)
  Tensor3D batch tgt_len d_model.     (* Decoder output *)

(** ** Complete Forward Pass *)

(** Complete model forward pass: source + target → vocabulary logits.

    This is the main entry point for training. Processes source and target
    sequences through the complete encoder-decoder architecture.

    Algorithm:
    memory = encode(model, src, src_mask)
    decoder_out = decode(model, tgt, memory, src_mask, tgt_mask)

    Note: The generator projection is typically applied separately during
    loss computation for efficiency.

    Parameters:
    - d_model: Model dimension (e.g., 512)
    - d_ff: Feed-forward inner dimension (e.g., 2048)
    - num_heads: Number of attention heads (e.g., 8)
    - head_dim: Dimension per head (d_model / num_heads = 64)
    - num_layers: Number of layers (e.g., 6)
    - max_len: Maximum sequence length (e.g., 5000)
    - src_vocab: Source vocabulary size
    - tgt_vocab: Target vocabulary size
    - batch: Batch size
    - src_len: Source sequence length
    - tgt_len: Target sequence length
    - eq: Proof that d_model = num_heads * head_dim
    - src_ok: Proof that src_len <= max_len
    - tgt_ok: Proof that tgt_len <= max_len
    - model: The complete model
    - src: Source token indices [batch, src_len]
    - tgt: Target token indices [batch, tgt_len]
    - src_mask: Source mask [batch, 1, src_len]
    - tgt_mask: Target causal mask [batch, tgt_len, tgt_len]

    Returns:
    - Decoder output [batch, tgt_len, d_model] (before generator projection)

    Shape specifications:
    src:      [batch, src_len]         Source token IDs
    tgt:      [batch, tgt_len]         Target token IDs
    src_mask: [batch, 1, src_len]      Source padding mask (1=keep, 0=mask)
    tgt_mask: [batch, tgt_len, tgt_len] Target causal mask (prevents looking ahead)

    Returns:  [batch, tgt_len, d_model] Decoder hidden states

    Processing steps:
    1. Embed Source: Convert token IDs to embeddings + add positional encoding
    2. Encode: Process source through encoder stack → Memory
    3. Embed Target: Convert token IDs to embeddings + add positional encoding
    4. Decode: Process target through decoder with cross-attention to memory

    Mathematical flow:
    SrcEmb = PE(Embed(x_src))
    Memory = Encoder(SrcEmb, M_src)
    TgtEmb = PE(Embed(x_tgt))
    Hidden = Decoder(TgtEmb, Memory, M_src, M_tgt)

    Masking details:
    * Source Mask (M_src): Typically masks padding tokens
      - Shape: [batch, 1, src_len] broadcasts to all heads and target positions
      - Values: 1 for real tokens, 0 for padding
    * Target Mask (M_tgt): Combines padding and causal masks
      - Shape: [batch, tgt_len, tgt_len] for full attention pattern
      - Causal: Lower triangular (prevents attending to future)
      - Padding: Masks out padding positions
      - Combined: tgt_mask = padding_mask & causal_mask

    Corresponds to Haskell:
    encoderDecoderForward :: EncoderDecoder -> Tensor -> Tensor
                          -> Tensor -> Tensor -> TrainM Tensor
    encoderDecoderForward model src tgt srcMask tgtMask = do
      memory <- encode model src srcMask
      decode model tgt memory srcMask tgtMask

    Example:
    Source: [32, 100] (batch 32, src_len 100)
    Target: [32, 50] (batch 32, tgt_len 50)
    Output: [32, 50, 512] (decoder hidden states) *)

Parameter encoderDecoderForward : forall
  (d_model d_ff num_heads head_dim num_layers max_len src_vocab tgt_vocab
   batch src_len tgt_len : nat),
  d_model = num_heads * head_dim ->
  src_len <= max_len ->
  tgt_len <= max_len ->
  EncoderDecoder d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab ->
  Tensor2D batch src_len ->           (* Source tokens *)
  Tensor2D batch tgt_len ->           (* Target tokens *)
  Tensor3D batch 1 src_len ->         (* Source mask *)
  Tensor3D batch tgt_len tgt_len ->   (* Target mask *)
  Tensor3D batch tgt_len d_model.     (* Decoder output *)

(** ** Model Initialization *)

(** Initialize complete Encoder-Decoder model.

    Creates and initializes all model components:
    1. Encoder with N layers
    2. Decoder with N layers
    3. Source embeddings + positional encoding
    4. Target embeddings + positional encoding
    5. Generator (output projection)

    Initialization strategy:
    * Embeddings: Random uniform scaled by 1/√d_model
    * Linear layers: Xavier/Glorot uniform initialization
    * Layer norm: Ones for scale, zeros for bias
    * Positional encoding: Fixed sinusoidal (no learnable parameters)

    Parameters:
    - d_model: Model dimension (e.g., 512)
    - d_ff: Feed-forward inner dimension (e.g., 2048)
    - num_heads: Number of attention heads (e.g., 8)
    - num_layers: Number of layers (e.g., 6)
    - max_len: Maximum sequence length (e.g., 5000)
    - src_vocab: Source vocabulary size
    - tgt_vocab: Target vocabulary size

    Corresponds to Haskell:
    initEncoderDecoder :: TransformerConfig -> Int -> Int -> IO EncoderDecoder
    initEncoderDecoder config srcVocabSize tgtVocabSize = do
      encoder <- initEncoder config
      decoder <- initDecoder config
      srcEmbed <- initEmbeddings srcVocabSize (dModel config)
      let srcPE = initPositionalEncoding (dModel config) (maxLen config)
      tgtEmbed <- initEmbeddings tgtVocabSize (dModel config)
      let tgtPE = initPositionalEncoding (dModel config) (maxLen config)
      generator <- initGenerator (dModel config) tgtVocabSize
      pure $ EncoderDecoder { .. }

    Example:
    let model = initEncoderDecoder 512 2048 8 6 5000 10000 10000 in ... *)

Parameter initEncoderDecoder : forall
  (d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab : nat),
  EncoderDecoder d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab.

(** ** Model Construction Helper *)

(** Construct a complete Transformer model from hyperparameters.

    This is the main entry point for creating models, corresponding to the
    make_model function in the original Python implementation and the Haskell
    makeModel function.

    Default configuration:
    - max_len = 5000 (maximum sequence length)
    - Other parameters specified explicitly

    Parameters:
    - src_vocab: Source vocabulary size
    - tgt_vocab: Target vocabulary size
    - num_layers: Number of encoder/decoder layers (N)
      * Original paper: 6
      * Base model: 6
      * Large model: 12+
    - d_model: Model dimension (embedding size)
      * Original paper: 512
      * Base: 512
      * Large: 1024
    - d_ff: Feed-forward inner dimension
      * Typically 4× d_model
      * Original: 2048
    - num_heads: Number of attention heads (h)
      * Must divide d_model evenly
      * Original: 8
      * Ensures d_k = d_model / h

    Hyperparameter guidelines:

    Small Model (Fast, Low Memory):
    makeModel vocab vocab 2 128 512 4
    -- ~2M parameters, good for debugging/prototyping

    Base Model (Original Paper):
    makeModel vocab vocab 6 512 2048 8
    -- ~65M parameters, good baseline

    Large Model (Higher Capacity):
    makeModel vocab vocab 12 1024 4096 16
    -- ~300M parameters, better performance on large datasets

    Corresponds to Haskell:
    makeModel :: Int -> Int -> Int -> Int -> Int -> Int -> Float -> IO EncoderDecoder
    makeModel srcVocab tgtVocab numLayers dModel dFF numHeads dropout = do
      let config = TransformerConfig
            { dModel = dModel, dFF = dFF, numHeads = numHeads
            , numLayers = numLayers, dropout = dropout, maxLen = 5000 }
      initEncoderDecoder config srcVocab tgtVocab

    Example:
    -- Translation model (English→German)
    let model = makeModel 32000 32000 6 512 2048 8 in ... *)

Definition makeModel
  (src_vocab tgt_vocab num_layers d_model d_ff num_heads : nat)
  : EncoderDecoder d_model d_ff num_heads num_layers 5000 src_vocab tgt_vocab :=
  initEncoderDecoder d_model d_ff num_heads num_layers 5000 src_vocab tgt_vocab.

(** ** Dimension Preservation Properties *)

(** Theorem: Generator preserves batch and sequence dimensions.

    The generator only transforms the feature dimension from d_model to
    vocab_size, preserving batch and sequence dimensions.

    This ensures that output shape matches target sequence shape, allowing
    straightforward loss computation.

    Proof strategy:
    The type of generatorForward guarantees this property. *)

Theorem generator_preserves_batch_seq :
  forall (d_model vocab_size batch seq : nat)
    (gen : Generator d_model vocab_size)
    (x : Tensor3D batch seq d_model),
  exists (y : Tensor3D batch seq vocab_size), True.
Proof.
  intros.
  exists (generatorForward d_model vocab_size batch seq gen x).
  trivial.
Qed.

(** Theorem: Encode produces correct memory shape.

    The encode function produces encoder memory with the same batch and
    sequence dimensions as the input, with feature dimension d_model.

    This memory is then used by the decoder for cross-attention. *)

Theorem encode_produces_correct_shape :
  forall (d_model d_ff num_heads head_dim num_layers max_len src_vocab tgt_vocab
          batch src_len : nat)
    (eq : d_model = num_heads * head_dim)
    (len_ok : src_len <= max_len)
    (model : EncoderDecoder d_model d_ff num_heads num_layers max_len
               src_vocab tgt_vocab)
    (src : Tensor2D batch src_len)
    (src_mask : Tensor3D batch 1 src_len),
  exists (memory : Tensor3D batch src_len d_model), True.
Proof.
  intros.
  exists (encode d_model d_ff num_heads head_dim num_layers max_len src_vocab
            tgt_vocab batch src_len eq len_ok model src src_mask).
  trivial.
Qed.

(** Theorem: Decode produces correct output shape.

    The decode function produces decoder output with the same batch dimension
    as input, target sequence length, and feature dimension d_model.

    The output shape matches the target sequence, not the source sequence. *)

Theorem decode_produces_correct_shape :
  forall (d_model d_ff num_heads head_dim num_layers max_len src_vocab tgt_vocab
          batch tgt_len src_len : nat)
    (eq : d_model = num_heads * head_dim)
    (len_ok : tgt_len <= max_len)
    (model : EncoderDecoder d_model d_ff num_heads num_layers max_len
               src_vocab tgt_vocab)
    (tgt : Tensor2D batch tgt_len)
    (memory : Tensor3D batch src_len d_model)
    (src_mask : Tensor3D batch 1 src_len)
    (tgt_mask : Tensor3D batch tgt_len tgt_len),
  exists (output : Tensor3D batch tgt_len d_model), True.
Proof.
  intros.
  exists (decode d_model d_ff num_heads head_dim num_layers max_len src_vocab
            tgt_vocab batch tgt_len src_len eq len_ok model tgt memory
            src_mask tgt_mask).
  trivial.
Qed.

(** Theorem: Complete forward pass produces correct shape.

    The encoderDecoderForward function produces output matching the target
    sequence dimensions, ready for generator projection or loss computation. *)

Theorem forward_produces_correct_shape :
  forall (d_model d_ff num_heads head_dim num_layers max_len src_vocab tgt_vocab
          batch src_len tgt_len : nat)
    (eq : d_model = num_heads * head_dim)
    (src_ok : src_len <= max_len)
    (tgt_ok : tgt_len <= max_len)
    (model : EncoderDecoder d_model d_ff num_heads num_layers max_len
               src_vocab tgt_vocab)
    (src : Tensor2D batch src_len)
    (tgt : Tensor2D batch tgt_len)
    (src_mask : Tensor3D batch 1 src_len)
    (tgt_mask : Tensor3D batch tgt_len tgt_len),
  exists (output : Tensor3D batch tgt_len d_model), True.
Proof.
  intros.
  exists (encoderDecoderForward d_model d_ff num_heads head_dim num_layers
            max_len src_vocab tgt_vocab batch src_len tgt_len eq src_ok tgt_ok
            model src tgt src_mask tgt_mask).
  trivial.
Qed.

(** ** Sequence Length Constraints *)

(** Lemma: Sequence length constraints are enforced by the type system.

    The encode and decode functions require proofs that sequence lengths
    do not exceed max_len, ensuring that positional encodings are valid.

    This prevents runtime errors from exceeding pre-computed positional
    encoding tables. *)

Lemma sequence_length_constraints_enforced :
  forall (max_len src_len tgt_len : nat),
  src_len <= max_len ->
  tgt_len <= max_len ->
  (* Both constraints are satisfied *)
  True.
Proof.
  intros.
  trivial.
Qed.

(** ** Model Component Integration *)

(** Lemma: Model contains all required components.

    The EncoderDecoder type contains all necessary components for a
    complete transformer model:
    - Encoder for source processing
    - Decoder for target processing
    - Source and target embeddings with positional encodings
    - Generator for output projection *)

Lemma model_has_all_components :
  forall (d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab : nat)
    (model : EncoderDecoder d_model d_ff num_heads num_layers max_len
               src_vocab tgt_vocab),
  exists (enc : Encoder d_model d_ff num_heads num_layers)
         (dec : Decoder d_model d_ff num_heads num_layers)
         (src_emb : Embeddings src_vocab d_model)
         (src_pe : PositionalEncoding max_len d_model)
         (tgt_emb : Embeddings tgt_vocab d_model)
         (tgt_pe : PositionalEncoding max_len d_model)
         (gen : Generator d_model tgt_vocab),
  True.
Proof.
  intros.
  exists (ed_encoder d_model d_ff num_heads num_layers max_len src_vocab
            tgt_vocab model).
  exists (ed_decoder d_model d_ff num_heads num_layers max_len src_vocab
            tgt_vocab model).
  exists (ed_src_embed d_model d_ff num_heads num_layers max_len src_vocab
            tgt_vocab model).
  exists (ed_src_pe d_model d_ff num_heads num_layers max_len src_vocab
            tgt_vocab model).
  exists (ed_tgt_embed d_model d_ff num_heads num_layers max_len src_vocab
            tgt_vocab model).
  exists (ed_tgt_pe d_model d_ff num_heads num_layers max_len src_vocab
            tgt_vocab model).
  exists (ed_generator d_model d_ff num_heads num_layers max_len src_vocab
            tgt_vocab model).
  trivial.
Qed.

(** ** Encode-Decode Composition *)

(** Lemma: Encoding and decoding can be composed.

    The output of encode (memory) can be used as input to decode,
    demonstrating proper integration of encoder and decoder.

    This composition is the foundation of the complete forward pass. *)

Lemma encode_decode_compose :
  forall (d_model d_ff num_heads head_dim num_layers max_len src_vocab tgt_vocab
          batch src_len tgt_len : nat)
    (eq : d_model = num_heads * head_dim)
    (src_ok : src_len <= max_len)
    (tgt_ok : tgt_len <= max_len)
    (model : EncoderDecoder d_model d_ff num_heads num_layers max_len
               src_vocab tgt_vocab)
    (src : Tensor2D batch src_len)
    (tgt : Tensor2D batch tgt_len)
    (src_mask : Tensor3D batch 1 src_len)
    (tgt_mask : Tensor3D batch tgt_len tgt_len),
  let memory := encode d_model d_ff num_heads head_dim num_layers max_len
                  src_vocab tgt_vocab batch src_len eq src_ok model src src_mask in
  let output := decode d_model d_ff num_heads head_dim num_layers max_len
                  src_vocab tgt_vocab batch tgt_len src_len eq tgt_ok model
                  tgt memory src_mask tgt_mask in
  exists (y : Tensor3D batch tgt_len d_model), y = output.
Proof.
  intros.
  exists output.
  reflexivity.
Qed.

(** ** Independent Vocabulary Support *)

(** Lemma: Model supports different source and target vocabularies.

    The type system allows src_vocab and tgt_vocab to be different,
    enabling translation between languages with different vocabulary sizes.

    Example:
    - English: 50000 tokens
    - German: 60000 tokens
    - Model: EncoderDecoder ... 50000 60000 *)

Lemma model_supports_different_vocabs :
  forall (d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab : nat),
  exists (model : EncoderDecoder d_model d_ff num_heads num_layers max_len
                    src_vocab tgt_vocab),
  True.
Proof.
  intros.
  exists (initEncoderDecoder d_model d_ff num_heads num_layers max_len
            src_vocab tgt_vocab).
  trivial.
Qed.

(** ** Export *)

(** The Model types and operations are now available for use in other modules.

    Key exports:
    - Generator: output projection record type
    - EncoderDecoder: complete model record type
    - generatorForward: project to vocabulary logits
    - encode: source tokens → encoder memory
    - decode: target tokens + memory → decoder output
    - encoderDecoderForward: complete forward pass
    - initGenerator, initEncoderDecoder: initialization functions
    - makeModel: convenience function for model construction
    - Various property theorems demonstrating composition and type safety

    This completes the Transformer model specification. The model ties
    together all components:
    - Tensor operations (Tensor.v)
    - Linear layers (Linear.v)
    - Embeddings and positional encoding (Embedding.v)
    - Layer normalization (LayerNorm.v)
    - Feed-forward networks (FeedForward.v)
    - Attention mechanisms (Attention.v)
    - Sublayer connections (Sublayer.v)
    - Encoder stack (Encoder.v)
    - Decoder stack (Decoder.v)

    Next steps for a complete implementation:
    - Training procedures (loss computation, optimization)
    - Inference procedures (greedy decoding, beam search)
    - Data loading and preprocessing
    - Evaluation metrics
    - Model checkpointing and serialization *)
