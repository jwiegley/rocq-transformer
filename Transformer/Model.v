(** * Complete Encoder-Decoder Transformer Model *)

(** The full model:
    src -> embedding -> positional -> encoder -> memory
    tgt -> embedding -> positional -> decoder -> generator -> logits

    Generator projects decoder output to vocabulary size. *)

From Transformer Require Import Tensor.
From Transformer Require Import Config.
From Transformer Require Import Linear.
From Transformer Require Import Encoder.
From Transformer Require Import Decoder.
From Transformer Require Import Embedding.
From Stdlib Require Import Init.Nat.
From Stdlib Require Import Lists.List.
Import ListNotations.

(** ** Generator *)

(** Projects d_model to vocab_size, then log-softmax. *)

Record Generator (d_model vocab_size : nat) := mkGenerator {
  genProj : Linear d_model vocab_size
}.

Arguments mkGenerator {d_model vocab_size}.
Arguments genProj {d_model vocab_size}.

Definition generatorForward {d_model vocab_size batch seq : nat}
  (gen : Generator d_model vocab_size)
  (x : Tensor3D batch seq d_model)
  : Tensor3D batch seq vocab_size :=
  let logits := linearForward (genProj gen) x in
  softmax3D batch seq vocab_size logits.

(** ** Full Encoder-Decoder Model *)

Record EncoderDecoder
  (d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab : nat) :=
mkEncoderDecoder {
  edEncoder : Encoder d_model d_ff num_heads num_layers;
  edDecoder : Decoder d_model d_ff num_heads num_layers;
  edSrcEmbed : Embeddings src_vocab d_model;
  edTgtEmbed : Embeddings tgt_vocab d_model;
  edSrcPos : PositionalEncoding max_len d_model;
  edTgtPos : PositionalEncoding max_len d_model;
  edGenerator : Generator d_model tgt_vocab
}.

Arguments mkEncoderDecoder {d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab}.
Arguments edEncoder {d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab}.
Arguments edDecoder {d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab}.
Arguments edSrcEmbed {d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab}.
Arguments edTgtEmbed {d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab}.
Arguments edSrcPos {d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab}.
Arguments edTgtPos {d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab}.
Arguments edGenerator {d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab}.

(** Encode source sequence. *)

Definition encode
  {d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab batch src_len : nat}
  (model : EncoderDecoder d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab)
  (src : Tensor2D batch src_len)
  (srcMask : option (Tensor3D batch src_len src_len))
  (srcLenProof : src_len <= max_len)
  : Tensor3D batch src_len d_model :=
  let embedded := embeddingsForward (edSrcEmbed model) src in
  let withPos := positionalEncodingForward (edSrcPos model) embedded srcLenProof in
  encoderForward (edEncoder model) withPos srcMask.

(** Decode given encoder memory. *)

Definition decode
  {d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab batch src_len tgt_len : nat}
  (model : EncoderDecoder d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab)
  (tgt : Tensor2D batch tgt_len)
  (memory : Tensor3D batch src_len d_model)
  (srcMask : option (Tensor3D batch tgt_len src_len))
  (tgtMask : option (Tensor3D batch tgt_len tgt_len))
  (tgtLenProof : tgt_len <= max_len)
  : Tensor3D batch tgt_len d_model :=
  let embedded := embeddingsForward (edTgtEmbed model) tgt in
  let withPos := positionalEncodingForward (edTgtPos model) embedded tgtLenProof in
  decoderForward (edDecoder model) withPos memory tgtMask srcMask.

(** Full forward pass: encode then decode. *)

Definition forward
  {d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab batch src_len tgt_len : nat}
  (model : EncoderDecoder d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab)
  (src : Tensor2D batch src_len)
  (tgt : Tensor2D batch tgt_len)
  (srcMask : option (Tensor3D batch src_len src_len))
  (tgtMask : option (Tensor3D batch tgt_len tgt_len))
  (crossMask : option (Tensor3D batch tgt_len src_len))
  (srcLenProof : src_len <= max_len)
  (tgtLenProof : tgt_len <= max_len)
  : Tensor3D batch tgt_len tgt_vocab :=
  let memory := encode model src srcMask srcLenProof in
  let decoded := decode model tgt memory crossMask tgtMask tgtLenProof in
  generatorForward (edGenerator model) decoded.

(** Initialization *)

Parameter initGenerator : forall (d_model vocab_size : nat),
  Generator d_model vocab_size.

Parameter initEncoderDecoder : forall
  (d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab : nat),
  EncoderDecoder d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab.

(** Properties *)

Lemma model_output_shape : forall
  (d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab batch src_len tgt_len : nat)
  (model : EncoderDecoder d_model d_ff num_heads num_layers max_len src_vocab tgt_vocab)
  (src : Tensor2D batch src_len)
  (tgt : Tensor2D batch tgt_len)
  (srcMask : option (Tensor3D batch src_len src_len))
  (tgtMask : option (Tensor3D batch tgt_len tgt_len))
  (crossMask : option (Tensor3D batch tgt_len src_len))
  (srcPf : src_len <= max_len)
  (tgtPf : tgt_len <= max_len),
  exists (y : Tensor3D batch tgt_len tgt_vocab), True.
Proof.
  intros.
  exists (forward model src tgt srcMask tgtMask crossMask srcPf tgtPf).
  trivial.
Qed.
