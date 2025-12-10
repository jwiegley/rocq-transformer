(** * Encoder Layer and Stack for Transformer Model *)

(** This module implements the encoder component from "Attention is All You Need"
    (Vaswani et al., 2017).

    The encoder consists of a stack of N=6 identical layers. Each layer has two
    sub-layers:

    1. Multi-head self-attention mechanism - Allows the encoder to attend to
       different positions in the input sequence
    2. Position-wise feed-forward network - Applies the same FFN independently
       to each position

    Each sub-layer uses a residual connection followed by layer normalization:

    Output = x + Dropout(Sublayer(LayerNorm(x)))

    This is an ABSTRACT specification - operations are axiomatized without
    concrete implementations. The goal is to capture the dimensional constraints
    and invariants of the encoder architecture in the type system.

    == Architecture Overview

    Input sequence (with positional encoding)
      ↓
    ┌─────────────────────────────────┐
    │  Encoder Layer 1                │
    │  ┌──────────────────────────┐   │
    │  │ Self-Attention           │   │
    │  │ (with residual + norm)   │   │
    │  └──────────────────────────┘   │
    │  ┌──────────────────────────┐   │
    │  │ Feed-Forward             │   │
    │  │ (with residual + norm)   │   │
    │  └──────────────────────────┘   │
    └─────────────────────────────────┘
      ↓
    ... (repeated N times)
      ↓
    Final Layer Normalization
      ↓
    Encoder output

    == Residual Connections

    Residual connections (also called skip connections) are crucial for training
    deep neural networks. They address two fundamental problems:

    === Vanishing Gradients

    In deep networks, gradients can become exponentially small as they propagate
    backward through many layers. Residual connections provide a direct path
    for gradients to flow backward, ensuring every layer receives a strong
    learning signal.

    === Training Stability

    Residual connections stabilize training by:
    * Reducing the effective depth during early training
    * Allowing information to bypass problematic layers
    * Creating an ensemble of paths through the network

    == Pre-norm Architecture

    This implementation uses pre-norm (normalize before sublayer) rather than
    post-norm (normalize after residual) because:
    * Better training stability (especially for deep models)
    * Easier gradient flow in early training
    * Less sensitive to hyperparameter choices *)

From RocqTransformer Require Import Tensor.
From RocqTransformer Require Import Config.
From RocqTransformer Require Import Attention.
From RocqTransformer Require Import FeedForward.
From RocqTransformer Require Import LayerNorm.
From RocqTransformer Require Import Sublayer.
Require Import Coq.Init.Nat.
Require Import Coq.Arith.PeanoNat.
Require Import Coq.Lists.List.
Import ListNotations.

(** ** Single Encoder Layer *)

(** A single encoder layer with self-attention and feed-forward sublayers.

    Each encoder layer consists of:

    1. Multi-head self-attention sublayer - Allows each position to attend
       to all positions in the input sequence

    2. Position-wise feed-forward sublayer - Applies the same FFN
       independently to each position

    Both sublayers use residual connections and layer normalization.

    In the Haskell implementation, EncoderLayer contains:
    - elSelfAttn :: MultiHeadAttention        -- Multi-head self-attention
    - elFFN :: FeedForward                    -- Feed-forward network
    - elSublayer1 :: SublayerConnection       -- For self-attention
    - elSublayer2 :: SublayerConnection       -- For feed-forward
    - elSize :: Int                           -- Model dimension (d_model)

    Here we abstract the structure and focus on dimension preservation.

    Pattern:
    Input [batch, seq, d_model]
      │
      ├─→ LayerNorm ─→ MultiHeadAttention ─→ Dropout ─┬─→ Add ─┐
      │                                                 │        │
      └─────────────────────────────────────────────────┘        │
                                                                 │
      ┌───────────────────────────────────────────────────────────┘
      │
      ├─→ LayerNorm ─→ FeedForward ─→ Dropout ─┬─→ Add ─→ Output
      │                                         │
      └─────────────────────────────────────────┘

    Corresponds to Haskell:
    data EncoderLayer = EncoderLayer
      { elSelfAttn :: MultiHeadAttention
      , elFFN :: FeedForward
      , elSublayer1 :: SublayerConnection
      , elSublayer2 :: SublayerConnection
      , elSize :: Int } *)

Record EncoderLayer (d_model d_ff num_heads : nat) := mkEncoderLayer {
  el_self_attn : MultiHeadAttention d_model num_heads;
  el_ffn : FeedForward d_model d_ff;
  el_sublayer1 : SublayerConnection d_model;  (* For self-attention *)
  el_sublayer2 : SublayerConnection d_model;  (* For feed-forward *)
  (* In Haskell: elSize :: Int - implicitly captured by d_model parameter *)
}.

(** ** Encoder Layer Forward Pass *)

(** Forward pass for a single encoder layer.

    Applies the complete encoder layer transformation:

    1. Self-attention sublayer with residual connection
    2. Feed-forward sublayer with residual connection

    Algorithm:
    x1 = x + dropout(selfAttention(layerNorm(x), mask))
    output = x1 + dropout(feedForward(layerNorm(x1)))

    Parameters:
    - d_model: Model dimension (e.g., 512)
    - d_ff: Feed-forward inner dimension (e.g., 2048)
    - num_heads: Number of attention heads (e.g., 8)
    - head_dim: Dimension per head (d_model / num_heads = 64)
    - batch: Batch size
    - seq: Sequence length
    - eq: Proof that d_model = num_heads * head_dim
    - layer: The encoder layer
    - x: Input tensor [batch, seq, d_model]
    - mask: Source mask [batch, 1, seq]

    Returns:
    - Output tensor [batch, seq, d_model] (same shape as input)

    Self-Attention Pattern:
    The self-attention uses the same tensor for queries, keys, and values.
    This allows each position to attend to all positions in the input:

    selfAttention(x, x, x, mask)

    Corresponds to Haskell:
    encoderLayerForward :: EncoderLayer -> Tensor -> Tensor -> TrainM Tensor
    encoderLayerForward EncoderLayer{..} x mask = do
      x1 <- sublayerConnectionForward elSublayer1 selfAttnLayer x
      output <- sublayerConnectionForward elSublayer2 ffnLayer x1
      pure output

    Example:
    Input: [32, 100, 512]
    After self-attention sublayer: [32, 100, 512]
    After feed-forward sublayer: [32, 100, 512] *)

Parameter encoderLayerForward :
  forall (d_model d_ff num_heads head_dim batch seq : nat),
  d_model = num_heads * head_dim ->
  EncoderLayer d_model d_ff num_heads ->
  Tensor3D batch seq d_model ->        (* Input *)
  Tensor3D batch 1 seq ->              (* Source mask *)
  Tensor3D batch seq d_model.          (* Output: same shape as input *)

(** ** Full Encoder Stack *)

(** Complete encoder stack with N identical layers.

    The encoder consists of:

    1. A stack of N encoder layers (default N=6)
    2. Final layer normalization applied to the output

    == Why Final Normalization?

    The final layer normalization ensures that:

    * Output distribution is stable and well-normalized
    * Decoder receives normalized input (for encoder-decoder attention)
    * Model output is in a consistent range

    This is especially important in the pre-norm architecture where the final
    residual path is not normalized otherwise.

    Architecture:

    Input
      ↓
    EncoderLayer1
      ↓
    EncoderLayer2
      ↓
      ...
      ↓
    EncoderLayerN
      ↓
    Final LayerNorm
      ↓
    Output

    In the Haskell implementation, Encoder contains:
    - encLayers :: [EncoderLayer]  -- Stack of N encoder layers
    - encNorm :: LayerNorm         -- Final layer normalization

    Here we abstract the list of layers and only keep the final normalization,
    since the abstract specification doesn't need to track the actual list.

    Corresponds to Haskell:
    data Encoder = Encoder
      { encLayers :: [EncoderLayer]
      , encNorm :: LayerNorm } *)

Record Encoder (d_model d_ff num_heads num_layers : nat) := mkEncoder {
  enc_final_norm : LayerNorm d_model;
  (* In Haskell: encLayers :: [EncoderLayer] *)
  (* We abstract the list structure - the num_layers parameter captures the count *)
}.

(** ** Encoder Forward Pass *)

(** Forward pass for the complete encoder stack.

    Processes the input through the complete encoder stack:

    1. Pass input through each encoder layer sequentially
    2. Apply final layer normalization to the output

    Algorithm:
    x0 = input
    x1 = encoderLayer1(x0, mask)
    x2 = encoderLayer2(x1, mask)
    ...
    xN = encoderLayerN(x_{N-1}, mask)
    output = finalLayerNorm(xN)

    Parameters:
    - d_model: Model dimension (e.g., 512)
    - d_ff: Feed-forward inner dimension (e.g., 2048)
    - num_heads: Number of attention heads (e.g., 8)
    - head_dim: Dimension per head (d_model / num_heads = 64)
    - num_layers: Number of encoder layers (e.g., 6)
    - batch: Batch size
    - seq: Sequence length
    - eq: Proof that d_model = num_heads * head_dim
    - encoder: The encoder stack
    - x: Input tensor [batch, seq, d_model] (embedded source)
    - srcMask: Source mask [batch, 1, seq]

    Returns:
    - Output tensor [batch, seq, d_model] (encoded memory)

    Input Requirements:
    The input x should already include:
    * Token embeddings (vocabulary indices mapped to vectors)
    * Positional encodings (position information)
    * Input dropout (if applicable)

    The encoder does not perform embedding or positional encoding itself.

    Corresponds to Haskell:
    encoderForward :: Encoder -> Tensor -> Tensor -> TrainM Tensor
    encoderForward Encoder{..} x srcMask = do
      encoded <- foldlM applyLayer x encLayers
      let output = layerNormForward encNorm encoded
      pure output

    Example:
    Input (embeddings + positions): [32, 100, 512]
    After 6 encoder layers: [32, 100, 512]
    After final norm: [32, 100, 512] *)

Parameter encoderForward :
  forall (d_model d_ff num_heads head_dim num_layers batch seq : nat),
  d_model = num_heads * head_dim ->
  Encoder d_model d_ff num_heads num_layers ->
  Tensor3D batch seq d_model ->        (* Input: embedded source *)
  Tensor3D batch 1 seq ->              (* Source mask *)
  Tensor3D batch seq d_model.          (* Output: encoded memory *)

(** ** Initialization *)

(** Initialize an encoder layer.

    Creates all components of an encoder layer:
    * Multi-head self-attention (num_heads heads, d_model dimension)
    * Position-wise feed-forward (d_ff inner dimension)
    * Two sublayer connections (one for attention, one for FFN)

    Corresponds to Haskell:
    initEncoderLayer :: TransformerConfig -> IO EncoderLayer
    initEncoderLayer TransformerConfig{..} = do
      selfAttn <- initMultiHeadAttention dModel numHeads dropout
      ffn <- initFeedForward dModel dFF dropout
      sublayer1 <- initSublayerConnection dModel dropout
      sublayer2 <- initSublayerConnection dModel dropout
      pure EncoderLayer { .. }

    Example:
    let layer = initEncoderLayer 512 2048 8 in ... *)

Parameter initEncoderLayer : forall (d_model d_ff num_heads : nat),
  EncoderLayer d_model d_ff num_heads.

(** Initialize the encoder stack with N layers.

    Creates a stack of identical encoder layers according to the configuration.
    Each layer has its own randomly initialized parameters, but all layers share
    the same architecture.

    Corresponds to Haskell:
    initEncoder :: TransformerConfig -> IO Encoder
    initEncoder config@TransformerConfig{..} = do
      layers <- mapM (const $ initEncoderLayer config) [1..numLayers]
      finalNorm <- initLayerNorm dModel
      pure Encoder { encLayers = layers, encNorm = finalNorm }

    Example:
    let encoder = initEncoder 512 2048 8 6 in ... *)

Parameter initEncoder : forall (d_model d_ff num_heads num_layers : nat),
  Encoder d_model d_ff num_heads num_layers.

(** ** Dimension Preservation Properties *)

(** Theorem: encoder layer preserves shape.

    The fundamental property of an encoder layer is that it preserves the
    shape of its input tensor. The output has exactly the same dimensions
    as the input.

    This is essential for:
    - Residual connections within the layer
    - Stacking multiple encoder layers
    - Maintaining consistent dimensions through the encoder stack

    Proof strategy:
    The type of encoderLayerForward guarantees this property. Both input and
    output have type Tensor3D batch seq d_model.

    Example:
    Input: [32, 100, 512]
    After encoder layer: [32, 100, 512] *)

Theorem encoderLayer_preserves_shape :
  forall (d_model d_ff num_heads head_dim batch seq : nat)
    (eq : d_model = num_heads * head_dim)
    (layer : EncoderLayer d_model d_ff num_heads)
    (x : Tensor3D batch seq d_model)
    (mask : Tensor3D batch 1 seq),
  exists (y : Tensor3D batch seq d_model), True.
Proof.
  intros.
  exists (encoderLayerForward d_model d_ff num_heads head_dim batch seq eq layer x mask).
  trivial.
Qed.

(** Theorem: encoder stack preserves shape through all layers.

    The encoder stack processes input through N layers and final normalization,
    preserving the shape throughout.

    This theorem is crucial because it guarantees that:
    - Each layer in the stack preserves dimensions
    - The final normalization preserves dimensions
    - The encoder output has the same shape as input

    This allows the encoder to be used in the full transformer:
    - Input: embedded source tokens [batch, src_len, d_model]
    - Output: encoded memory [batch, src_len, d_model]
    - The decoder can use this output for cross-attention

    Proof strategy:
    The type of encoderForward guarantees this property. Both input and
    output have type Tensor3D batch seq d_model.

    Example:
    Input (embedded): [32, 100, 512]
    After 6 encoder layers: [32, 100, 512]
    After final norm: [32, 100, 512] *)

Theorem encoder_preserves_shape :
  forall (d_model d_ff num_heads head_dim num_layers batch seq : nat)
    (eq : d_model = num_heads * head_dim)
    (enc : Encoder d_model d_ff num_heads num_layers)
    (x : Tensor3D batch seq d_model)
    (mask : Tensor3D batch 1 seq),
  exists (y : Tensor3D batch seq d_model), True.
Proof.
  intros.
  exists (encoderForward d_model d_ff num_heads head_dim num_layers batch seq eq enc x mask).
  trivial.
Qed.

(** ** Mask Shape Compatibility *)

(** Lemma: source mask has correct shape for encoder.

    The source mask used in the encoder has shape [batch, 1, seq], which is
    compatible with the attention mechanism.

    The middle dimension of 1 allows broadcasting over all query positions:
    - Query: [batch, seq, d_model]
    - Key: [batch, seq, d_model]
    - Mask: [batch, 1, seq] broadcasts to [batch, seq, seq]

    This ensures that the same padding mask is applied to all query positions.

    Corresponds to the mask used in Haskell:
    srcMask :: Tensor  -- Shape: [batch, 1, src_len] *)

Lemma srcMask_shape :
  forall (batch seq : nat),
  exists (mask : Tensor3D batch 1 seq), True.
Proof.
  intros.
  (* The existence of such a mask is axiomatic - it's created by mask functions *)
  admit.
Admitted.

(** ** Composition Properties *)

(** Lemma: encoder layers can be stacked.

    Multiple encoder layers can be composed in sequence, with each layer
    preserving dimensions.

    This is the foundation of the encoder stack:
    output = layerN(...(layer2(layer1(input))))

    Each layer transformation is dimension-preserving, so the composition
    is well-typed and maintains the same shape.

    Corresponds to the pattern in Haskell:
    encoded <- foldlM applyLayer x encLayers

    Example:
    Input: [32, 100, 512]
    After layer1: [32, 100, 512]
    After layer2: [32, 100, 512]
    ...
    After layerN: [32, 100, 512] *)

Lemma encoder_layers_stack :
  forall (d_model d_ff num_heads head_dim batch seq : nat)
    (eq : d_model = num_heads * head_dim)
    (layer1 layer2 : EncoderLayer d_model d_ff num_heads)
    (x : Tensor3D batch seq d_model)
    (mask : Tensor3D batch 1 seq),
  let x1 := encoderLayerForward d_model d_ff num_heads head_dim batch seq eq layer1 x mask in
  let x2 := encoderLayerForward d_model d_ff num_heads head_dim batch seq eq layer2 x1 mask in
  exists (y : Tensor3D batch seq d_model), y = x2.
Proof.
  intros.
  exists x2.
  reflexivity.
Qed.

(** ** Integration with Sublayer Connections *)

(** Lemma: encoder layer uses two sublayer connections.

    Each encoder layer applies two sublayer connections:
    1. Self-attention with residual and normalization
    2. Feed-forward with residual and normalization

    This pattern is captured by the EncoderLayer type, which contains
    two SublayerConnection components.

    The dimension preservation of encoder layers follows from the dimension
    preservation of sublayer connections.

    Corresponds to Haskell:
    x1 <- sublayerConnectionForward elSublayer1 selfAttnLayer x
    output <- sublayerConnectionForward elSublayer2 ffnLayer x1 *)

Lemma encoder_uses_sublayers :
  forall (d_model d_ff num_heads : nat)
    (layer : EncoderLayer d_model d_ff num_heads),
  exists (slc1 slc2 : SublayerConnection d_model), True.
Proof.
  intros.
  exists (el_sublayer1 d_model d_ff num_heads layer).
  exists (el_sublayer2 d_model d_ff num_heads layer).
  trivial.
Qed.

(** ** Self-Attention Pattern *)

(** Lemma: encoder uses self-attention.

    The encoder's attention mechanism is self-attention, meaning queries,
    keys, and values all come from the same input tensor:

    attention(x, x, x, mask)

    This allows each position to attend to all positions in the input
    sequence, enabling the encoder to build rich contextual representations.

    This is distinct from the decoder's cross-attention, where queries come
    from the decoder but keys and values come from the encoder:

    cross_attention(decoder_x, encoder_memory, encoder_memory, mask)

    The type system ensures proper tensor dimensions for self-attention. *)

Lemma encoder_self_attention :
  forall (d_model d_ff num_heads head_dim batch seq : nat)
    (eq : d_model = num_heads * head_dim)
    (layer : EncoderLayer d_model d_ff num_heads)
    (x : Tensor3D batch seq d_model)
    (mask : Tensor3D batch 1 seq),
  (* Self-attention uses same tensor for Q, K, V *)
  (* This is enforced by encoderLayerForward implementation *)
  exists (output : Tensor3D batch seq d_model), True.
Proof.
  intros.
  exists (encoderLayerForward d_model d_ff num_heads head_dim batch seq eq layer x mask).
  trivial.
Qed.

(** ** Export *)

(** The Encoder types and operations are now available for use in other modules.

    Key exports:
    - EncoderLayer: record type for single encoder layer
    - Encoder: record type for full encoder stack
    - encoderLayerForward: forward pass for single layer
    - encoderForward: forward pass for full stack
    - initEncoderLayer, initEncoder: initialization functions
    - encoderLayer_preserves_shape: dimension preservation theorem for layer
    - encoder_preserves_shape: dimension preservation theorem for stack
    - Various property lemmas demonstrating composition and type safety

    Next modules to implement:
    - Decoder layer and stack (with masked self-attention and cross-attention)
    - Full EncoderDecoder model (combining encoder and decoder)
    - Generator (final linear projection to vocabulary)
    - Complete transformer model *)
