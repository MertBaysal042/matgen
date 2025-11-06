# MatGen - Parallel Sparse Matrix Scaling and Value Estimation

A high-performance C library for generating sparse matrices through parallel scaling algorithms (Nearest Neighbor and Bilinear Interpolation) with realistic value estimation. Implements OpenMP, MPI, and CUDA backends for scalability.

## 📚 Research Foundation

This project implements algorithms from the following research:

1. **Agarwal, A., Dahleh, M., Shah, D., and Shen, D. (2021)**
   _Causal Matrix Completion_
   arXiv:2109.15154v1
   - Matrix completion techniques for sparse matrices
   - Value estimation and prediction methods

2. **Bruch, S., Nardini, F. M., Rulli, C., and Venturini, R. (2025)**
   _Efficient Sketching and Nearest Neighbor Search Algorithms for Sparse Vector Sets_
   arXiv:2509.24815v1
   - Efficient nearest neighbor algorithms for sparse data
   - Distance computations and similarity metrics

3. **MatGen Framework by Ali Emre Pamuk**
   - Sparse matrix generation methodology

## 🎯 Project Goals

Generate new sparse matrices by **scaling existing ones** using two interpolation methods:

### 1. Nearest Neighbor Scaling

- Maps each output position to its nearest input position
- Preserves exact sparsity (output has at most input nnz)
- Fast, simple, suitable for discrete data
- Parallel implementations: OpenMP, MPI, CUDA

### 2. Bilinear Interpolation Scaling

- Weighted average of 4 neighboring input positions
- Smooth interpolation with sparsity control
- Can densify output (controlled by threshold)
- Parallel implementations: OpenMP, MPI, CUDA

### 3. Realistic Value Estimation

- Predict values for newly interpolated positions
- Learn distributions from real sparse matrices
- Maintain statistical properties during scaling

### 4. Extensions

- Non-square (rectangular) matrix scaling
- Parallel structural feature extraction
- Matrix quality validation

## 🏗️ Project Structure

```
matgen/
├── src/
│   ├── core/                    # Core sparse matrix formats
│   │   ├── coo.c               # Coordinate format
│   │   ├── csr.c               # Compressed Sparse Row
│   │   └── conversion.c        # Format conversion
│   │
│   ├── io/                      # Input/Output
│   │   ├── mtx_reader.c        # Matrix Market reader
│   │   └── mtx_writer.c        # Matrix Market writer
│   │
│   ├── scaling/                 # Scaling algorithms (sequential)
│   │   ├── nearest.c           # Nearest neighbor scaling
│   │   └── bilinear.c          # Bilinear interpolation scaling
│   │
│   ├── parallel/                # Parallel implementations
│   │   ├── openmp/             # OpenMP backend
│   │   │   ├── scale_nearest_omp.c
│   │   │   ├── scale_bilinear_omp.c
│   │   │   └── omp_utils.c
│   │   ├── mpi/                # MPI backend
│   │   │   ├── scale_nearest_mpi.c
│   │   │   ├── scale_bilinear_mpi.c
│   │   │   └── mpi_utils.c
│   │   └── cuda/               # CUDA backend
│   │       ├── scale_nearest.cu
│   │       ├── scale_bilinear.cu
│   │       └── cuda_utils.cu
│   │
│   ├── values/                  # Value estimation
│   │   ├── value_learner.c     # Learn from real matrices
│   │   ├── value_estimator.c   # Estimate during scaling
│   │   └── distributions.c     # Statistical distributions
│   │
│   ├── features/                # Feature extraction
│   │   ├── degree_dist.c       # Degree distribution
│   │   ├── statistics.c        # Matrix statistics
│   │   └── quality_metrics.c   # Quality evaluation
│   │
│   └── ops/                     # Matrix operations
│       ├── spmv.c              # Sparse matrix-vector multiply
│       ├── vector_ops.c        # Dense vector operations
│       └── distances.c         # Sparse distance metrics
│
├── include/matgen/              # Public API headers
├── tests/                       # Unit tests (GoogleTest)
├── benchmarks/                  # Performance benchmarks
└── examples/                    # Usage examples
```

## 🚀 Implementation Status

### ✅ Completed

- [x] COO and CSR sparse matrix formats
- [x] Format conversion (COO ↔ CSR)
- [x] Matrix Market I/O with symmetric expansion
- [x] Sequential nearest neighbor scaling
- [x] Sequential bilinear interpolation scaling
- [x] OpenMP parallel scaling (both methods)
- [x] MPI distributed scaling (broadcast-gather strategy)
- [] CUDA GPU scaling (dynamic sparse output)
- [x] Matrix operations (SpMV, vector ops, distances)

### 🔄 In Progress

- [ ] Testing and validation of all parallel backends
- [ ] Performance benchmarking (strong/weak scaling)
- [ ] CUDA kernel optimization (two-pass algorithm)

### 📋 Planned

- [ ] Value estimation implementation
  - [ ] Learn value distributions from real matrices
  - [ ] Statistical models (normal, log-normal, power-law)
  - [ ] Value prediction during interpolation
- [ ] Non-square matrix support
- [ ] Parallel feature extraction
- [ ] Matrix quality metrics
- [ ] Comprehensive benchmarking suite

## 🛠️ Building the Project

### Requirements

- **C Compiler**: GCC 9.0+ or Clang 10.0+ (C17)
- **CMake**: 3.25+
- **Ninja**: Recommended build system

**Optional:**

- **OpenMP**: 4.5+ (multi-core parallelism)
- **MPI**: OpenMPI 4.0+ or MPICH 3.3+ (distributed)
- **CUDA**: 11.0+ (GPU acceleration)
- **GoogleTest**: Unit testing
- **GoogleBenchmark**: Performance testing

### Quick Build

```bash
# Configure
cmake --preset windows-msvc-release

# Build
cmake --build --preset windows-msvc-release

# Test
ctest --preset windows-msvc-release
```

### Enable Parallel Backends

```bash
# OpenMP only
cmake --preset windows-msvc-release -DENABLE_OPENMP=ON

# MPI only
cmake --preset windows-msvc-release -DENABLE_MPI=ON

# CUDA only
cmake --preset windows-msvc-release -DENABLE_CUDA=ON

# All backends
cmake --preset windows-msvc-release -DENABLE_OPENMP=ON -DENABLE_MPI=ON -DENABLE_CUDA=ON
```

## 📊 Usage Examples

### Scale Matrix with Nearest Neighbor (Sequential)

```cpp
#include <matgen/io/mtx_reader.h>
#include <matgen/io/mtx_writer.h>
#include <matgen/scaling/nearest.h>

// Read input matrix (1000x1000)
MatGenCOO* input = matgen_mtx_read_coo("input.mtx");

// Scale to 5000x5000 using nearest neighbor
MatGenCOO* output = matgen_scale_nearest(input, 5000, 5000);

// Write output
matgen_mtx_write_coo("output.mtx", output);

// Cleanup
matgen_coo_destroy(input);
matgen_coo_destroy(output);
```

### Scale with Bilinear Interpolation (OpenMP)

```cpp
#include <matgen/parallel/openmp/scale_bilinear_omp.h>

// Convert to CSR for efficient lookup
MatGenCSR* input_csr = matgen_coo_to_csr(input);

// Scale with 8 threads, sparsity threshold = 0.01
omp_set_num_threads(8);
MatGenCOO* output = matgen_scale_bilinear_omp(input_csr, 5000, 5000, 0.01);

matgen_csr_destroy(input_csr);
```

### Distributed Scaling with MPI

```cpp
#include <matgen/parallel/mpi/scale_nearest_mpi.h>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Root loads input matrix
    MatGenCOO* input = NULL;
    if (rank == 0) {
        input = matgen_mtx_read_coo("input.mtx");
    }

    // Distributed scaling
    MatGenCOO* output = matgen_scale_nearest_mpi(input, 10000, 10000,
                                                   MPI_COMM_WORLD);

    // Root writes result
    if (rank == 0) {
        matgen_mtx_write_coo("output.mtx", output);
    }

    matgen_coo_destroy(input);
    matgen_coo_destroy(output);

    MPI_Finalize();
    return 0;
}
```

### GPU Scaling with CUDA

```cpp
#include <matgen/parallel/cuda/scale_bilinear.h>

MatGenCSR* input_csr = matgen_coo_to_csr(input);

// Scale on GPU
MatGenCOO* output = matgen_scale_bilinear_cuda(input_csr, 5000, 5000, 0.01);

matgen_csr_destroy(input_csr);
```

## 📈 Performance Goals

- **OpenMP**: Near-linear speedup up to number of physical cores
- **MPI**: Scalability to 100+ processes for large matrices
- **CUDA**: 10-100x speedup over sequential for large matrices

## 🧪 Testing

```bash
# Run all tests
ctest --preset windows-msvc-release -V

# Run specific test suite
./out/build/windows-msvc-release/tests/test_scaling
```
