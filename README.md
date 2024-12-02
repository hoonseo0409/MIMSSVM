# MIMSSVM: Multi-Instance Multi-Shape Support Vector Machine

## Overview
Implementation of Multi-Instance Multi-Shape Support Vector Machine (MIMSSVM), a novel approach for multi-instance learning that effectively handles image instances which consist of patches with the varying shapes. The model integrates group sparsity regularization and trace norm optimization to capture multi-modal patterns in the data.

## Requirements
- Julia 1.6 or later
- Required packages are specified in Project.toml

## Repository Structure
1. `MIMSSVMClassifier.jl`: Core implementation of MIMSSVM
   - Implements both exact and inexact optimization strategies
   - Supports multi-modal feature learning through structured sparsity
   - Includes kernel extension for non-linear decision boundaries
   - Optimizes using multi-block ADMM algorithm

2. `mimssvmclassifier_test.jl`: Test suite that verifies:
   - Variable updates with zero derivatives
   - Objective function decreases at each update step
   - Convergence of optimization algorithm
   - Correctness of prediction outputs

3. `Project.toml`: Project dependencies including:
   - Core ML frameworks: MLJ, Flux
   - Optimization libraries
   - Data processing utilities
   - Visualization tools

## Installation & Usage

1. Start Julia in command line interface:
```julia
julia
```

2. Set up project environment:
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

3. Run tests:
```julia
include("path-to-mimssvmclassifier_test.jl")
```

## Model Parameters
- `M_cut`: Specifies modality groups for multi-shape learning
- `δ`: Smoothness parameter for regularization (default: 1e-10)
- `τ_1`: Group sparsity regularization strength (default: 1.0)
- `τ_2`: Trace norm regularization strength (default: 1.0)
- `C`: Classification cost parameter (default: 1.0)
- `exact`: Boolean flag for exact/inexact optimization (default: true)

## Features
- Multi-modal learning capability
- Structured sparsity regularization
- Both exact and inexact optimization strategies
- Kernel extension for non-linear classification
- GPU acceleration support via CUDA.jl