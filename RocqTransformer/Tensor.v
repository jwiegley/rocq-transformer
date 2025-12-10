(** * Abstract Tensor Type System for Transformer Model *)

(** This module defines dimension-indexed tensor types and abstract operations
    for the Rocq Transformer implementation. The tensor types are parameterized
    by their dimensions at the type level, enabling compile-time dimension safety.

    This is an ABSTRACT specification - operations are axiomatized without
    concrete implementations. The goal is to capture dimension constraints
    in the type system for verification purposes, not to perform actual
    tensor computations. *)

Require Import Coq.Init.Nat.
Require Import Coq.Lists.List.
Require Import Coq.Arith.PeanoNat.
Import ListNotations.

(** ** Type-Level Dimension Specifications *)

(** We use lists of natural numbers to represent tensor dimensions.
    For example:
    - [rows; cols] represents a 2D matrix
    - [batch; rows; cols] represents a 3D batched tensor
    - [batch; heads; rows; cols] represents a 4D multi-head tensor *)

Definition DimSpec := list nat.

(** ** Core Tensor Type *)

(** The main tensor type is indexed by its dimension specification.
    This allows the type system to track dimensions through operations. *)

Inductive Tensor : DimSpec -> Type :=
  | tensor : forall (dims : DimSpec), Tensor dims.

(** ** Convenience Type Aliases *)

(** These aliases match the Haskell implementation's Tensor2D, Tensor3D, Tensor4D
    constructors, making the code more readable and maintaining consistency with
    the reference implementation. *)

Definition Tensor2D (rows cols : nat) : Type :=
  Tensor [rows; cols].

Definition Tensor3D (batch rows cols : nat) : Type :=
  Tensor [batch; rows; cols].

Definition Tensor4D (batch heads rows cols : nat) : Type :=
  Tensor [batch; heads; rows; cols].

(** ** Matrix Multiplication Operations *)

(** Matrix multiplication for 2D tensors with dimension constraints.
    For A : Tensor2D m k and B : Tensor2D k n, produces C : Tensor2D m n.
    The inner dimensions must match (both k). *)

Parameter matmul2D : forall (m n k : nat),
  Tensor2D m k -> Tensor2D k n -> Tensor2D m n.

(** Batched matrix multiplication for 3D tensors.
    Applies matmul independently to each batch element.
    A : [batch, m, k] x B : [batch, k, n] -> C : [batch, m, n] *)

Parameter matmul3D : forall (batch m n k : nat),
  Tensor3D batch m k -> Tensor3D batch k n -> Tensor3D batch m n.

(** 4D matrix multiplication for multi-head attention.
    Each head in each batch performs independent matmul.
    A : [batch, heads, m, k] x B : [batch, heads, k, n]
    -> C : [batch, heads, m, n] *)

Parameter matmul4D : forall (batch heads m n k : nat),
  Tensor4D batch heads m k ->
  Tensor4D batch heads k n ->
  Tensor4D batch heads m n.

(** ** Transpose Operations *)

(** Transpose a 2D matrix: swap rows and columns.
    [m, n] -> [n, m] *)

Parameter transpose2D : forall (m n : nat),
  Tensor2D m n -> Tensor2D n m.

(** Transpose the last two dimensions of a 3D tensor.
    Each matrix in the batch is transposed independently.
    [batch, m, n] -> [batch, n, m] *)

Parameter transpose3D_12 : forall (batch m n : nat),
  Tensor3D batch m n -> Tensor3D batch n m.

(** Transpose the last two dimensions of a 4D tensor.
    Each matrix in each head of each batch is transposed.
    [batch, heads, m, n] -> [batch, heads, n, m] *)

Parameter transpose4D_23 : forall (batch heads m n : nat),
  Tensor4D batch heads m n -> Tensor4D batch heads n m.

(** Transpose batch and heads dimensions of a 4D tensor.
    [batch, heads, m, n] -> [heads, batch, m, n] *)

Parameter transpose4D_01 : forall (batch heads m n : nat),
  Tensor4D batch heads m n -> Tensor4D heads batch m n.

(** ** Shape-Preserving Operations *)

(** These operations maintain the same dimensions as their input. *)

(** Softmax: normalize along a dimension.
    Commonly applied to attention scores. *)

Parameter softmax : forall (dims : DimSpec),
  Tensor dims -> Tensor dims.

(** ReLU activation: max(0, x).
    Standard activation function for feed-forward networks. *)

Parameter relu : forall (dims : DimSpec),
  Tensor dims -> Tensor dims.

(** Layer normalization: normalize features with mean and variance.
    Essential for training stability in transformers. *)

Parameter layerNorm : forall (dims : DimSpec),
  Tensor dims -> Tensor dims.

(** Dropout: randomly zero elements during training.
    Used for regularization. *)

Parameter dropout : forall (dims : DimSpec),
  Tensor dims -> Tensor dims.

(** Element-wise addition: add corresponding elements.
    Used for residual connections: output = input + sublayer(input). *)

Parameter add : forall (dims : DimSpec),
  Tensor dims -> Tensor dims -> Tensor dims.

(** Scalar multiplication: multiply every element by a scalar.
    Used for attention scaling: scores / sqrt(d_k). *)

Parameter scale : forall (dims : DimSpec),
  Tensor dims -> Tensor dims.

(** Element-wise multiplication.
    Used in dropout masking and gating mechanisms. *)

Parameter mul : forall (dims : DimSpec),
  Tensor dims -> Tensor dims -> Tensor dims.

(** Element-wise subtraction.
    Used in normalization computations. *)

Parameter sub : forall (dims : DimSpec),
  Tensor dims -> Tensor dims -> Tensor dims.

(** ** Reshape Operations *)

(** General view/reshape operation.
    Changes dimensions while preserving total number of elements.
    In a real implementation, would require proof that
    product(dims1) = product(dims2). *)

Parameter view : forall (dims1 dims2 : DimSpec),
  Tensor dims1 -> Tensor dims2.

(** Specific reshape for multi-head attention:
    [batch, seq, d_model] -> [batch, seq, heads, d_k]
    where d_model = heads * d_k *)

Parameter viewToHeads : forall (batch seq heads d_k : nat),
  Tensor3D batch seq (heads * d_k) ->
  Tensor4D batch seq heads d_k.

(** Reshape from multi-head format back to single dimension:
    [batch, seq, heads, d_k] -> [batch, seq, d_model]
    where d_model = heads * d_k *)

Parameter viewFromHeads : forall (batch seq heads d_k : nat),
  Tensor4D batch seq heads d_k ->
  Tensor3D batch seq (heads * d_k).

(** ** Concatenation Operations *)

(** Concatenate along batch dimension.
    Two 3D tensors with matching row/col dimensions but different batch sizes
    combine into a single tensor with batch size = b1 + b2. *)

Parameter cat_batch : forall (b1 b2 m n : nat),
  Tensor3D b1 m n -> Tensor3D b2 m n -> Tensor3D (b1 + b2) m n.

(** Concatenate along sequence dimension.
    Useful for combining source and target sequences. *)

Parameter cat_seq : forall (batch s1 s2 d : nat),
  Tensor3D batch s1 d -> Tensor3D batch s2 d -> Tensor3D batch (s1 + s2) d.

(** ** Masking Operations *)

(** Masked fill: set elements to a value where mask is true.
    Essential for attention masking (padding and causal masks). *)

Parameter maskedFill : forall (dims : DimSpec),
  Tensor dims -> (* mask *)
  Tensor dims -> (* values *)
  Tensor dims.   (* result *)

(** Create causal mask for decoder self-attention.
    Result is lower-triangular: position i can attend to positions <= i.
    Returns a [seq_len, seq_len] mask. *)

Parameter subsequentMask : forall (seq_len : nat),
  Tensor2D seq_len seq_len.

(** Create padding mask from sequence lengths.
    Masks out padding tokens (beyond actual sequence length). *)

Parameter paddingMask : forall (batch max_len : nat),
  Tensor2D batch max_len.

(** ** Selection and Indexing *)

(** Select a specific index along a dimension.
    Used to extract specific batch elements or sequence positions. *)

Parameter select : forall (dims : DimSpec) (idx : nat),
  Tensor dims -> Tensor dims.

(** Gather values along the last dimension using indices.
    Used for embedding lookup: indices -> embedding vectors. *)

Parameter gather : forall (dims idx_dims : DimSpec),
  Tensor dims -> Tensor idx_dims -> Tensor idx_dims.

(** Argmax: find the index of maximum value along a dimension.
    Used for getting predicted tokens in generation. *)

Parameter argmax : forall (dims : DimSpec),
  Tensor dims -> Tensor dims.

(** ** Creation Operations *)

(** Create tensor filled with zeros. *)

Parameter zeros : forall (dims : DimSpec),
  Tensor dims.

(** Create tensor filled with ones. *)

Parameter ones : forall (dims : DimSpec),
  Tensor dims.

(** Create tensor with random values from normal distribution.
    Used for weight initialization. *)

Parameter randn : forall (dims : DimSpec),
  Tensor dims.

(** ** Utility Functions *)

(** Get the shape (dimensions) of a tensor as a list. *)

Parameter shape : forall (dims : DimSpec),
  Tensor dims -> DimSpec.

(** Get the size of a specific dimension. *)

Parameter size : forall (dims : DimSpec) (dim : nat),
  Tensor dims -> nat.

(** Extract scalar value from a single-element tensor. *)

Parameter item : forall (dims : DimSpec),
  Tensor dims -> nat.

(** ** Embedding Operations *)

(** Embedding lookup: convert token indices to embedding vectors.
    weights: [vocab_size, d_model]
    indices: [batch, seq_len]
    result:  [batch, seq_len, d_model] *)

Parameter embeddingLookup : forall (vocab_size d_model batch seq_len : nat),
  Tensor2D vocab_size d_model ->
  Tensor2D batch seq_len ->
  Tensor3D batch seq_len d_model.

(** ** Basic Dimension Preservation Properties *)

(** These are examples of properties we might want to prove about tensor operations.
    They express invariants that the type system enforces. *)

(** Matmul produces the expected output dimensions *)

Axiom matmul2D_dims : forall (m n k : nat) (A : Tensor2D m k) (B : Tensor2D k n),
  shape _ (matmul2D m n k A B) = [m; n].

(** Transpose swaps dimensions correctly *)

Axiom transpose2D_dims : forall (m n : nat) (A : Tensor2D m n),
  shape _ (transpose2D m n A) = [n; m].

(** Addition preserves dimensions *)

Axiom add_preserves_dims : forall (dims : DimSpec) (A B : Tensor dims),
  shape _ (add dims A B) = dims.

(** Transpose is involutive (transposing twice returns original) *)

Axiom transpose2D_involutive : forall (m n : nat) (A : Tensor2D m n),
  transpose2D n m (transpose2D m n A) = A.

(** Matmul is associative when dimensions align *)

Axiom matmul2D_assoc : forall (m n k p : nat)
  (A : Tensor2D m k) (B : Tensor2D k n) (C : Tensor2D n p),
  matmul2D m p n (matmul2D m n k A B) C =
  matmul2D m p k A (matmul2D k p n B C).

(** ** Module Export *)

(** Make the tensor types and operations available for use in other modules. *)
