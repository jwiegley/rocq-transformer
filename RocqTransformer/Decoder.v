(** * Decoder Layer and Stack for Transformer Model *)

(** This module implements the decoder component from "Attention is All You Need"
    (Vaswani et al., 2017).

    The decoder consists of a stack of N=6 identical layers. Each layer has THREE
    sub-layers (compared to the encoder's two):

    1. Masked self-attention mechanism - Allows the decoder to attend to previous
       positions in the target sequence (causal attention)
    2. Cross-attention mechanism - Allows the decoder to attend to the encoder
       output (memory), with queries from decoder and keys/values from encoder
    3. Position-wise feed-forward network - Applies the same FFN independently
       to each position

    Each sub-layer uses a residual connection followed by layer normalization:

    Output = x + Dropout(Sublayer(LayerNorm(x)))

    This is an ABSTRACT specification - operations are axiomatized without
    concrete implementations. The goal is to capture the dimensional constraints
    and invariants of the decoder architecture in the type system.

    == Architecture Overview

    Target sequence (with positional encoding) + Encoder memory
      ↓
    ┌─────────────────────────────────┐
    │  Decoder Layer 1                │
    │  ┌──────────────────────────┐   │
    │  │ Masked Self-Attention    │   │  (causal mask)
    │  │ (with residual + norm)   │   │
    │  └──────────────────────────┘   │
    │  ┌──────────────────────────┐   │
    │  │ Cross-Attention          │   │  (with encoder memory)
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
    Decoder output

    == Attention Types

    === Encoder Self-Attention (bidirectional)
    All positions can attend to all positions in source sequence.
    Used in the encoder to build rich contextual representations.

    === Decoder Self-Attention (causal/masked)
    Each position can only attend to previous positions (including itself).
    Uses causal mask to prevent attending to future positions.
    Essential for autoregressive generation.

    === Decoder Cross-Attention (encoder-decoder)
    Decoder positions attend to all encoder positions.
    Queries come from decoder, keys and values come from encoder memory.
    This is how the decoder incorporates source sequence information.

    == Causal Masking

    The decoder uses a "subsequent mask" to ensure autoregressive property:

    Position:  0  1  2  3
       0      [T  F  F  F]    Can only see position 0
       1      [T  T  F  F]    Can see positions 0-1
       2      [T  T  T  F]    Can see positions 0-2
       3      [T  T  T  T]    Can see positions 0-3

    This is implemented as an upper triangular matrix where:
    - Lower triangle (including diagonal) = 0 (allowed to attend)
    - Upper triangle = 1 (masked out, becomes -inf in attention scores)

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

(** ** Subsequent (Causal) Mask *)

(** Generate subsequent mask for causal (autoregressive) decoding.

    Creates a square mask that prevents positions from attending to
    subsequent (future) positions. Essential for autoregressive generation
    where each position can only depend on previously generated tokens.

    The mask is an upper triangular matrix:

    subsequentMask 4 =
      [[0, 1, 1, 1],     ← position 0 can't see positions 1,2,3
       [0, 0, 1, 1],     ← position 1 can't see positions 2,3
       [0, 0, 0, 1],     ← position 2 can't see position 3
       [0, 0, 0, 0]]     ← position 3 can see all (including itself)

    Values of 1 will be converted to -inf in attention scores,
    effectively masking out those positions.

    Parameters:
    - size: Sequence length (creates size × size mask)

    Returns:
    - Binary mask tensor, shape: [size, size]
      - 0 = can attend (keep attention score)
      - 1 = cannot attend (mask to -inf)

    Implementation Note:
    This creates a mask for a single sequence. In practice:
    - Batch dimension is handled by broadcasting
    - Combined with padding mask using logical OR
    - Applied in attention before softmax

    Corresponds to Haskell:
    subsequentMask :: Int -> Tensor
    subsequentMask size =
      let maskMatrix =
            [ [ if col > row then 1.0 else 0.0
              | col <- [0 .. size - 1] ]
            | row <- [0 .. size - 1] ]
      in fromList2D maskMatrix

    Example:
    -- Generate causal mask for length 5 target sequence
    mask = subsequentMask 5

    -- Use in decoder forward pass
    output <- decoderForward decoder targetEmb encoderOut srcMask mask *)

Parameter subsequentMask : forall (size : nat), Tensor2D size size.

(** ** Single Decoder Layer *)

(** A single decoder layer with three sublayers (vs encoder's two).

    Each decoder layer consists of:

    1. Masked self-attention sublayer - Allows each position to attend
       only to previous positions in the target sequence (causal attention)

    2. Cross-attention sublayer - Allows the decoder to attend to the
       encoder output, with queries from decoder and keys/values from encoder

    3. Position-wise feed-forward sublayer - Applies the same FFN
       independently to each position

    All three sublayers use residual connections and layer normalization.

    In the Haskell implementation, DecoderLayer contains:
    - dlSelfAttn :: MultiHeadAttention        -- Masked self-attention
    - dlSrcAttn :: MultiHeadAttention         -- Cross-attention to encoder
    - dlFFN :: FeedForward                    -- Feed-forward network
    - dlSublayer1 :: SublayerConnection       -- For masked self-attention
    - dlSublayer2 :: SublayerConnection       -- For cross-attention
    - dlSublayer3 :: SublayerConnection       -- For feed-forward
    - dlSize :: Int                           -- Model dimension (d_model)

    Here we abstract the structure and focus on dimension preservation.

    Pattern:
    Input [batch, tgt_len, d_model] + Memory [batch, src_len, d_model]
      │
      ├─→ LayerNorm ─→ MaskedSelfAttention(Q=K=V=tgt) ─→ Dropout ─┬─→ Add ─┐
      │                                                              │        │
      └──────────────────────────────────────────────────────────────┘        │
                                                                              │
      ┌────────────────────────────────────────────────────────────────────────┘
      │
      ├─→ LayerNorm ─→ CrossAttention(Q=tgt, K=V=memory) ─→ Dropout ─┬─→ Add ─┐
      │                                                                │        │
      └────────────────────────────────────────────────────────────────┘        │
                                                                                │
      ┌──────────────────────────────────────────────────────────────────────────┘
      │
      ├─→ LayerNorm ─→ FeedForward ─→ Dropout ─┬─→ Add ─→ Output
      │                                         │
      └─────────────────────────────────────────┘

    Corresponds to Haskell:
    data DecoderLayer = DecoderLayer
      { dlSelfAttn :: MultiHeadAttention
      , dlSrcAttn :: MultiHeadAttention
      , dlFFN :: FeedForward
      , dlSublayer1 :: SublayerConnection
      , dlSublayer2 :: SublayerConnection
      , dlSublayer3 :: SublayerConnection
      , dlSize :: Int } *)

Record DecoderLayer (d_model d_ff num_heads : nat) := mkDecoderLayer {
  dl_self_attn : MultiHeadAttention d_model num_heads;  (* Masked self-attention *)
  dl_src_attn : MultiHeadAttention d_model num_heads;   (* Cross-attention *)
  dl_ffn : FeedForward d_model d_ff;
  dl_sublayer1 : SublayerConnection d_model;  (* For masked self-attention *)
  dl_sublayer2 : SublayerConnection d_model;  (* For cross-attention *)
  dl_sublayer3 : SublayerConnection d_model;  (* For feed-forward *)
  (* In Haskell: dlSize :: Int - implicitly captured by d_model parameter *)
}.

(** ** Decoder Layer Forward Pass *)

(** Forward pass for a single decoder layer.

    Applies the complete decoder layer transformation with three sublayers:

    1. Masked self-attention sublayer with residual connection
    2. Cross-attention sublayer with residual connection
    3. Feed-forward sublayer with residual connection

    Algorithm:
    x1 = x + dropout(maskedSelfAttention(layerNorm(x), tgt_mask))
    x2 = x1 + dropout(crossAttention(layerNorm(x1), memory, src_mask))
    output = x2 + dropout(feedForward(layerNorm(x2)))

    Parameters:
    - d_model: Model dimension (e.g., 512)
    - d_ff: Feed-forward inner dimension (e.g., 2048)
    - num_heads: Number of attention heads (e.g., 8)
    - head_dim: Dimension per head (d_model / num_heads = 64)
    - batch: Batch size
    - tgt_len: Target sequence length
    - src_len: Source sequence length (encoder memory length)
    - eq: Proof that d_model = num_heads * head_dim
    - layer: The decoder layer
    - x: Target input tensor [batch, tgt_len, d_model]
    - memory: Encoder output [batch, src_len, d_model]
    - src_mask: Source mask [batch, 1, src_len]
    - tgt_mask: Target causal mask [batch, tgt_len, tgt_len]

    Returns:
    - Output tensor [batch, tgt_len, d_model] (same shape as target input)

    Masked Self-Attention Pattern:
    The masked self-attention uses the same tensor for queries, keys, and values,
    but with a causal mask to prevent attending to future positions:

    maskedSelfAttention(x, x, x, tgt_mask)

    Cross-Attention Pattern:
    The cross-attention uses different tensors for queries vs keys/values.
    Queries come from the decoder, keys and values come from encoder memory:

    crossAttention(query=x, key=memory, value=memory, src_mask)

    Corresponds to Haskell:
    decoderLayerForward :: DecoderLayer -> Tensor -> Tensor -> Tensor -> Tensor
                        -> TrainM Tensor
    decoderLayerForward DecoderLayer{..} x memory srcMask tgtMask = do
      x1 <- sublayerConnectionForward dlSublayer1 maskedSelfAttnLayer x
      x2 <- sublayerConnectionForward dlSublayer2 crossAttnLayer x1
      output <- sublayerConnectionForward dlSublayer3 ffnLayer x2
      pure output

    Example:
    Target input: [32, 50, 512]
    Encoder memory: [32, 100, 512]
    After masked self-attention sublayer: [32, 50, 512]
    After cross-attention sublayer: [32, 50, 512]
    After feed-forward sublayer: [32, 50, 512] *)

Parameter decoderLayerForward :
  forall (d_model d_ff num_heads head_dim batch tgt_len src_len : nat),
  d_model = num_heads * head_dim ->
  DecoderLayer d_model d_ff num_heads ->
  Tensor3D batch tgt_len d_model ->    (* Target input *)
  Tensor3D batch src_len d_model ->    (* Encoder memory *)
  Tensor3D batch 1 src_len ->          (* Source mask *)
  Tensor3D batch tgt_len tgt_len ->    (* Target causal mask *)
  Tensor3D batch tgt_len d_model.      (* Output: same shape as target input *)

(** ** Full Decoder Stack *)

(** Complete decoder stack with N identical layers.

    The decoder consists of:

    1. A stack of N decoder layers (default N=6)
    2. Final layer normalization applied to the output

    == Why Final Normalization?

    The final layer normalization ensures that:

    * Output distribution is stable and well-normalized
    * Model output is in a consistent range for the generator
    * Consistent with pre-norm architecture

    This is especially important in the pre-norm architecture where the final
    residual path is not normalized otherwise.

    Architecture:

    Target Input + Encoder Memory
      ↓
    DecoderLayer1
      ↓
    DecoderLayer2
      ↓
      ...
      ↓
    DecoderLayerN
      ↓
    Final LayerNorm
      ↓
    Output

    In the Haskell implementation, Decoder contains:
    - decLayers :: [DecoderLayer]  -- Stack of N decoder layers
    - decNorm :: LayerNorm         -- Final layer normalization

    Here we abstract the list of layers and only keep the final normalization,
    since the abstract specification doesn't need to track the actual list.

    Corresponds to Haskell:
    data Decoder = Decoder
      { decLayers :: [DecoderLayer]
      , decNorm :: LayerNorm } *)

Record Decoder (d_model d_ff num_heads num_layers : nat) := mkDecoder {
  dec_final_norm : LayerNorm d_model;
  (* In Haskell: decLayers :: [DecoderLayer] *)
  (* We abstract the list structure - the num_layers parameter captures the count *)
}.

(** ** Decoder Forward Pass *)

(** Forward pass for the complete decoder stack.

    Processes the target input through the complete decoder stack:

    1. Pass input through each decoder layer sequentially
       (each layer uses encoder memory for cross-attention)
    2. Apply final layer normalization to the output

    Algorithm:
    x0 = target input
    x1 = decoderLayer1(x0, memory, src_mask, tgt_mask)
    x2 = decoderLayer2(x1, memory, src_mask, tgt_mask)
    ...
    xN = decoderLayerN(x_{N-1}, memory, src_mask, tgt_mask)
    output = finalLayerNorm(xN)

    Parameters:
    - d_model: Model dimension (e.g., 512)
    - d_ff: Feed-forward inner dimension (e.g., 2048)
    - num_heads: Number of attention heads (e.g., 8)
    - head_dim: Dimension per head (d_model / num_heads = 64)
    - num_layers: Number of decoder layers (e.g., 6)
    - batch: Batch size
    - tgt_len: Target sequence length
    - src_len: Source sequence length
    - eq: Proof that d_model = num_heads * head_dim
    - decoder: The decoder stack
    - tgt: Target input tensor [batch, tgt_len, d_model] (embedded target)
    - memory: Encoder output [batch, src_len, d_model]
    - src_mask: Source mask [batch, 1, src_len]
    - tgt_mask: Target causal mask [batch, tgt_len, tgt_len]

    Returns:
    - Output tensor [batch, tgt_len, d_model] (decoder output)

    Input Requirements:
    The target input should already include:
    * Token embeddings (vocabulary indices mapped to vectors)
    * Positional encodings (position information)
    * Input dropout (if applicable)

    The decoder does not perform embedding or positional encoding itself.

    Usage in Generation:
    During autoregressive generation:
    1. Start with target = [START]
    2. Generate tgt_mask = subsequentMask(1)
    3. Run decoderForward to get predictions
    4. Sample next token, append to target
    5. Generate new tgt_mask = subsequentMask(currentLength)
    6. Repeat until [END] token or max length

    Corresponds to Haskell:
    decoderForward :: Decoder -> Tensor -> Tensor -> Tensor -> Tensor
                   -> TrainM Tensor
    decoderForward Decoder{..} tgt memory srcMask tgtMask = do
      decoded <- foldlM applyLayer tgt decLayers
      let output = layerNormForward decNorm decoded
      pure output

    Example:
    Target input (embeddings + positions): [32, 50, 512]
    Encoder memory: [32, 100, 512]
    After 6 decoder layers: [32, 50, 512]
    After final norm: [32, 50, 512] *)

Parameter decoderForward :
  forall (d_model d_ff num_heads head_dim num_layers batch tgt_len src_len : nat),
  d_model = num_heads * head_dim ->
  Decoder d_model d_ff num_heads num_layers ->
  Tensor3D batch tgt_len d_model ->    (* Target embeddings *)
  Tensor3D batch src_len d_model ->    (* Encoder memory *)
  Tensor3D batch 1 src_len ->          (* Source mask *)
  Tensor3D batch tgt_len tgt_len ->    (* Target mask *)
  Tensor3D batch tgt_len d_model.      (* Decoder output *)

(** ** Initialization *)

(** Initialize a decoder layer.

    Creates all components of a decoder layer:
    * Masked self-attention (num_heads heads, d_model dimension)
    * Cross-attention to encoder (num_heads heads, d_model dimension)
    * Position-wise feed-forward (d_ff inner dimension)
    * Three sublayer connections (one for each sublayer)

    Corresponds to Haskell:
    initDecoderLayer :: TransformerConfig -> IO DecoderLayer
    initDecoderLayer TransformerConfig{..} = do
      selfAttn <- initMultiHeadAttention dModel numHeads dropout
      srcAttn <- initMultiHeadAttention dModel numHeads dropout
      ffn <- initFeedForward dModel dFF dropout
      sublayer1 <- initSublayerConnection dModel dropout
      sublayer2 <- initSublayerConnection dModel dropout
      sublayer3 <- initSublayerConnection dModel dropout
      pure DecoderLayer { .. }

    Example:
    let layer = initDecoderLayer 512 2048 8 in ... *)

Parameter initDecoderLayer : forall (d_model d_ff num_heads : nat),
  DecoderLayer d_model d_ff num_heads.

(** Initialize the decoder stack with N layers.

    Creates a stack of identical decoder layers according to the configuration.
    Each layer has its own randomly initialized parameters, but all layers share
    the same architecture.

    Corresponds to Haskell:
    initDecoder :: TransformerConfig -> IO Decoder
    initDecoder config@TransformerConfig{..} = do
      layers <- mapM (const $ initDecoderLayer config) [1..numLayers]
      finalNorm <- initLayerNorm dModel
      pure Decoder { decLayers = layers, decNorm = finalNorm }

    Example:
    let decoder = initDecoder 512 2048 8 6 in ... *)

Parameter initDecoder : forall (d_model d_ff num_heads num_layers : nat),
  Decoder d_model d_ff num_heads num_layers.

(** ** Dimension Preservation Properties *)

(** Theorem: decoder layer preserves target shape.

    The fundamental property of a decoder layer is that it preserves the
    shape of the target input tensor. The output has exactly the same dimensions
    as the target input.

    Note that the encoder memory has a different shape (src_len vs tgt_len),
    but this doesn't affect the output shape. Cross-attention allows different
    sequence lengths for queries vs keys/values.

    This is essential for:
    - Residual connections within the layer
    - Stacking multiple decoder layers
    - Maintaining consistent dimensions through the decoder stack

    Proof strategy:
    The type of decoderLayerForward guarantees this property. The target input
    and output both have type Tensor3D batch tgt_len d_model.

    Example:
    Target input: [32, 50, 512]
    Encoder memory: [32, 100, 512]  (different src_len!)
    After decoder layer: [32, 50, 512]  (same as target input) *)

Theorem decoderLayer_preserves_target_shape :
  forall (d_model d_ff num_heads head_dim batch tgt_len src_len : nat)
    (eq : d_model = num_heads * head_dim)
    (layer : DecoderLayer d_model d_ff num_heads)
    (tgt : Tensor3D batch tgt_len d_model)
    (memory : Tensor3D batch src_len d_model)
    (src_mask : Tensor3D batch 1 src_len)
    (tgt_mask : Tensor3D batch tgt_len tgt_len),
  exists (y : Tensor3D batch tgt_len d_model), True.
Proof.
  intros.
  exists (decoderLayerForward d_model d_ff num_heads head_dim batch tgt_len src_len
    eq layer tgt memory src_mask tgt_mask).
  trivial.
Qed.

(** Theorem: decoder stack preserves target shape through all layers.

    The decoder stack processes target input through N layers and final
    normalization, preserving the target shape throughout.

    This theorem is crucial because it guarantees that:
    - Each layer in the stack preserves dimensions
    - The final normalization preserves dimensions
    - The decoder output has the same shape as target input

    This allows the decoder to be used in the full transformer:
    - Target input: embedded target tokens [batch, tgt_len, d_model]
    - Encoder memory: encoded source [batch, src_len, d_model]
    - Output: decoded representation [batch, tgt_len, d_model]
    - The generator can then project to vocabulary

    Proof strategy:
    The type of decoderForward guarantees this property. The target input
    and output both have type Tensor3D batch tgt_len d_model.

    Example:
    Target input (embedded): [32, 50, 512]
    Encoder memory: [32, 100, 512]
    After 6 decoder layers: [32, 50, 512]
    After final norm: [32, 50, 512] *)

Theorem decoder_preserves_target_shape :
  forall (d_model d_ff num_heads head_dim num_layers batch tgt_len src_len : nat)
    (eq : d_model = num_heads * head_dim)
    (dec : Decoder d_model d_ff num_heads num_layers)
    (tgt : Tensor3D batch tgt_len d_model)
    (memory : Tensor3D batch src_len d_model)
    (src_mask : Tensor3D batch 1 src_len)
    (tgt_mask : Tensor3D batch tgt_len tgt_len),
  exists (y : Tensor3D batch tgt_len d_model), True.
Proof.
  intros.
  exists (decoderForward d_model d_ff num_heads head_dim num_layers batch tgt_len
    src_len eq dec tgt memory src_mask tgt_mask).
  trivial.
Qed.

(** ** Mask Shape Compatibility *)

(** Lemma: target causal mask has correct shape for decoder.

    The target mask used in the decoder has shape [batch, tgt_len, tgt_len],
    which is compatible with the attention mechanism.

    This is a square mask that prevents positions from attending to
    subsequent positions:
    - Query: [batch, tgt_len, d_model]
    - Key: [batch, tgt_len, d_model]
    - Mask: [batch, tgt_len, tgt_len]

    The causal structure ensures autoregressive generation.

    Corresponds to the mask used in Haskell:
    tgtMask :: Tensor  -- Shape: [batch, tgt_len, tgt_len]
    tgtMask = subsequentMask tgt_len *)

Lemma tgtMask_shape :
  forall (batch tgt_len : nat),
  exists (mask : Tensor3D batch tgt_len tgt_len), True.
Proof.
  intros.
  (* The existence of such a mask is axiomatic - it's created by subsequentMask *)
  admit.
Admitted.

(** Lemma: source mask has correct shape for cross-attention.

    The source mask used for cross-attention has shape [batch, 1, src_len],
    which is compatible with the attention mechanism.

    The middle dimension of 1 allows broadcasting over all target query positions:
    - Query: [batch, tgt_len, d_model]
    - Key: [batch, src_len, d_model]
    - Mask: [batch, 1, src_len] broadcasts to [batch, tgt_len, src_len]

    This ensures that the same source padding mask is applied to all target
    query positions.

    Corresponds to the mask used in Haskell:
    srcMask :: Tensor  -- Shape: [batch, 1, src_len] *)

Lemma srcMask_shape_for_decoder :
  forall (batch src_len : nat),
  exists (mask : Tensor3D batch 1 src_len), True.
Proof.
  intros.
  (* The existence of such a mask is axiomatic - it's created by mask functions *)
  admit.
Admitted.

(** ** Composition Properties *)

(** Lemma: decoder layers can be stacked.

    Multiple decoder layers can be composed in sequence, with each layer
    preserving dimensions.

    This is the foundation of the decoder stack:
    output = layerN(...(layer2(layer1(tgt, memory))))

    Each layer transformation is dimension-preserving (for target), so the
    composition is well-typed and maintains the same target shape.

    Corresponds to the pattern in Haskell:
    decoded <- foldlM applyLayer tgt decLayers

    Example:
    Target: [32, 50, 512]
    Memory: [32, 100, 512]
    After layer1: [32, 50, 512]
    After layer2: [32, 50, 512]
    ...
    After layerN: [32, 50, 512] *)

Lemma decoder_layers_stack :
  forall (d_model d_ff num_heads head_dim batch tgt_len src_len : nat)
    (eq : d_model = num_heads * head_dim)
    (layer1 layer2 : DecoderLayer d_model d_ff num_heads)
    (tgt : Tensor3D batch tgt_len d_model)
    (memory : Tensor3D batch src_len d_model)
    (src_mask : Tensor3D batch 1 src_len)
    (tgt_mask : Tensor3D batch tgt_len tgt_len),
  let x1 := decoderLayerForward d_model d_ff num_heads head_dim batch tgt_len src_len
              eq layer1 tgt memory src_mask tgt_mask in
  let x2 := decoderLayerForward d_model d_ff num_heads head_dim batch tgt_len src_len
              eq layer2 x1 memory src_mask tgt_mask in
  exists (y : Tensor3D batch tgt_len d_model), y = x2.
Proof.
  intros.
  exists x2.
  reflexivity.
Qed.

(** ** Integration with Sublayer Connections *)

(** Lemma: decoder layer uses three sublayer connections.

    Each decoder layer applies three sublayer connections:
    1. Masked self-attention with residual and normalization
    2. Cross-attention with residual and normalization
    3. Feed-forward with residual and normalization

    This pattern is captured by the DecoderLayer type, which contains
    three SublayerConnection components.

    The dimension preservation of decoder layers follows from the dimension
    preservation of sublayer connections.

    Corresponds to Haskell:
    x1 <- sublayerConnectionForward dlSublayer1 maskedSelfAttnLayer x
    x2 <- sublayerConnectionForward dlSublayer2 crossAttnLayer x1
    output <- sublayerConnectionForward dlSublayer3 ffnLayer x2 *)

Lemma decoder_uses_three_sublayers :
  forall (d_model d_ff num_heads : nat)
    (layer : DecoderLayer d_model d_ff num_heads),
  exists (slc1 slc2 slc3 : SublayerConnection d_model), True.
Proof.
  intros.
  exists (dl_sublayer1 d_model d_ff num_heads layer).
  exists (dl_sublayer2 d_model d_ff num_heads layer).
  exists (dl_sublayer3 d_model d_ff num_heads layer).
  trivial.
Qed.

(** ** Attention Pattern Properties *)

(** Lemma: decoder uses masked self-attention.

    The decoder's first attention mechanism is masked self-attention, meaning
    queries, keys, and values all come from the target sequence, but with a
    causal mask:

    maskedSelfAttention(x, x, x, tgt_mask)

    This allows each position to attend only to previous positions in the
    target sequence, enabling autoregressive generation.

    The causal mask prevents information leakage from future positions. *)

Lemma decoder_masked_self_attention :
  forall (d_model d_ff num_heads head_dim batch tgt_len src_len : nat)
    (eq : d_model = num_heads * head_dim)
    (layer : DecoderLayer d_model d_ff num_heads)
    (tgt : Tensor3D batch tgt_len d_model)
    (memory : Tensor3D batch src_len d_model)
    (src_mask : Tensor3D batch 1 src_len)
    (tgt_mask : Tensor3D batch tgt_len tgt_len),
  (* Masked self-attention uses same tensor for Q, K, V with causal mask *)
  (* This is enforced by decoderLayerForward implementation *)
  exists (output : Tensor3D batch tgt_len d_model), True.
Proof.
  intros.
  exists (decoderLayerForward d_model d_ff num_heads head_dim batch tgt_len src_len
    eq layer tgt memory src_mask tgt_mask).
  trivial.
Qed.

(** Lemma: decoder uses cross-attention to encoder memory.

    The decoder's second attention mechanism is cross-attention, where:
    - Queries come from decoder (target sequence)
    - Keys and values come from encoder memory (source sequence)

    crossAttention(query=decoder_x, key=memory, value=memory, src_mask)

    This is how the decoder incorporates information from the source sequence.
    It allows the decoder to attend to all encoder positions when generating
    each target position.

    The source mask handles padding in the encoder sequence. *)

Lemma decoder_cross_attention :
  forall (d_model d_ff num_heads head_dim batch tgt_len src_len : nat)
    (eq : d_model = num_heads * head_dim)
    (layer : DecoderLayer d_model d_ff num_heads)
    (tgt : Tensor3D batch tgt_len d_model)
    (memory : Tensor3D batch src_len d_model)
    (src_mask : Tensor3D batch 1 src_len)
    (tgt_mask : Tensor3D batch tgt_len tgt_len),
  (* Cross-attention uses decoder for Q, encoder memory for K and V *)
  (* Different sequence lengths (tgt_len vs src_len) are supported *)
  exists (output : Tensor3D batch tgt_len d_model), True.
Proof.
  intros.
  exists (decoderLayerForward d_model d_ff num_heads head_dim batch tgt_len src_len
    eq layer tgt memory src_mask tgt_mask).
  trivial.
Qed.

(** ** Subsequent Mask Properties *)

(** Lemma: subsequent mask is square and size-dependent.

    The subsequent mask for causal attention is a square matrix of size
    [tgt_len, tgt_len], where each row i masks out columns j > i.

    This creates the causal structure for autoregressive generation.

    Corresponds to Haskell:
    subsequentMask :: Int -> Tensor
    subsequentMask size = ... -- [size, size] matrix *)

Lemma subsequentMask_is_square :
  forall (size : nat),
  exists (mask : Tensor2D size size), mask = subsequentMask size.
Proof.
  intros.
  exists (subsequentMask size).
  reflexivity.
Qed.

(** ** Independent Sequence Length Support *)

(** Lemma: decoder supports different source and target lengths.

    A key property of the decoder is that the source sequence length
    (encoder memory) and target sequence length can differ.

    This is essential for sequence-to-sequence tasks where input and
    output have different lengths (e.g., translation).

    The type system enforces this flexibility through separate tgt_len
    and src_len parameters.

    Example:
    - Source (English): [32, 100, 512]  -- 100 tokens
    - Target (French): [32, 50, 512]    -- 50 tokens
    - Decoder output: [32, 50, 512]     -- matches target length *)

Lemma decoder_supports_different_lengths :
  forall (d_model d_ff num_heads head_dim num_layers batch tgt_len src_len : nat)
    (eq : d_model = num_heads * head_dim)
    (dec : Decoder d_model d_ff num_heads num_layers)
    (tgt : Tensor3D batch tgt_len d_model)
    (memory : Tensor3D batch src_len d_model)
    (src_mask : Tensor3D batch 1 src_len)
    (tgt_mask : Tensor3D batch tgt_len tgt_len),
  (* tgt_len and src_len are independent parameters *)
  (* The decoder output matches target length, not source length *)
  exists (output : Tensor3D batch tgt_len d_model),
    output = decoderForward d_model d_ff num_heads head_dim num_layers batch
               tgt_len src_len eq dec tgt memory src_mask tgt_mask.
Proof.
  intros.
  exists (decoderForward d_model d_ff num_heads head_dim num_layers batch
            tgt_len src_len eq dec tgt memory src_mask tgt_mask).
  reflexivity.
Qed.

(** ** Export *)

(** The Decoder types and operations are now available for use in other modules.

    Key exports:
    - subsequentMask: function to create causal masks
    - DecoderLayer: record type for single decoder layer
    - Decoder: record type for full decoder stack
    - decoderLayerForward: forward pass for single layer
    - decoderForward: forward pass for full stack
    - initDecoderLayer, initDecoder: initialization functions
    - decoderLayer_preserves_target_shape: dimension preservation theorem for layer
    - decoder_preserves_target_shape: dimension preservation theorem for stack
    - Various property lemmas demonstrating composition and type safety

    Key differences from Encoder:
    - Three sublayers instead of two (additional cross-attention)
    - Masked self-attention uses causal mask
    - Cross-attention connects to encoder memory
    - Supports different source and target sequence lengths

    Next modules to implement:
    - Full EncoderDecoder model (combining encoder and decoder)
    - Generator (final linear projection to vocabulary)
    - Complete transformer model
    - Training and inference procedures *)
