# Rocq Transformer Implementation

This directory contains a formally verified implementation of the Transformer model from "Attention is All You Need" (Vaswani et al., 2017) using Rocq (formerly Coq 8.18+).

## Project Structure

```
rocq/
├── _CoqProject           # Coq project configuration
├── Makefile              # Build configuration (generated)
├── RocqTransformer/      # Main module directory
│   ├── Tensor.v          # Tensor types and operations
│   ├── Config.v          # Model configuration
│   └── ...               # Additional modules
└── README.md             # This file
```

## Building

### Prerequisites

- Rocq/Coq 8.18 or newer
- Standard library (Reals, Lists)

### Using Nix (Recommended)

```bash
nix develop                          # Enter dev shell with Rocq
cd rocq/
coq_makefile -f _CoqProject -o Makefile
make                                 # Build all modules
```

### Manual Build

```bash
cd rocq/
coq_makefile -f _CoqProject -o Makefile
make all
```

## Modules

- **Tensor.v**: Defines tensor types and operations for matrix computations
- **Config.v**: Transformer configuration parameters matching the paper defaults

## Development

The Rocq implementation follows the Haskell implementation in `../haskell/` but adds formal proofs of correctness for key operations.

### Module Organization

The module namespace is `RocqTransformer`, configured via the `_CoqProject` file with:
```
-Q . RocqTransformer
```

### Compilation

Individual modules can be compiled:
```bash
make RocqTransformer/Tensor.vo
make RocqTransformer/Config.vo
```

Clean compiled files:
```bash
make clean
```

## Integration with Haskell

This Rocq implementation serves as a formal specification and verification layer for the Haskell implementation. The types and operations mirror the Haskell modules in `../haskell/src/Transformer/`.

## References

- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017
- [Rocq Documentation](https://coq.inria.fr/doc/)
- [The Annotated Transformer (Haskell)](../haskell/README.md)
