# MatGen - Parallel Sparse Matrix Scaling and Value Estimation

A high-performance C library for generating sparse matrices through parallel scaling algorithms (Nearest Neighbor and Bilinear Interpolation) with realistic value estimation. Implements OpenMP, MPI, and CUDA backends for scalability.

## Features

- **Multiple Scaling Algorithms**

  - Nearest Neighbor (value preservation)
  - Bilinear Interpolation (smooth value distribution)

- **Multi-Backend Support**

  - Sequential (baseline)
  - OpenMP (shared-memory parallelism)
  - MPI (distributed-memory parallelism)
  - CUDA (GPU acceleration)

- **Sparse Matrix Formats**
  - CSR (Compressed Sparse Row)
  - COO (Coordinate format)
  - Matrix Market (.mtx) I/O

## Build Requirements

### Required

- **CMake** 3.28 or later
- **C Compiler** with C17 support (MSVC 2019+, GCC 9+, Clang 10+)
- **C++ Compiler** with C++20 support (for tests)
- **Ninja** build system (recommended)

### Optional (Backend-Specific)

- **OpenMP** - For shared-memory parallelization (enabled by default)

  - LLVM OpenMP runtime on Windows (libomp)
  - Built-in on GCC/Clang Linux

- **MPI** - For distributed-memory parallelization (enabled by default)

  - Microsoft MPI (Windows): [Download](https://www.microsoft.com/en-us/download/details.aspx?id=105289)
  - OpenMPI (Linux): `sudo apt install libopenmpi-dev`
  - MPICH (Linux): `sudo apt install libmpich-dev`

- **CUDA Toolkit** 11.0+ - For GPU acceleration (enabled by default)
  - [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
  - Requires NVIDIA GPU with Compute Capability 6.0+

### Testing (Optional)

- **Google Test** (bundled as submodule)

## Building on Windows

### Prerequisites

1. **Install Visual Studio 2022** with:

- Desktop Development with C++
- MSVC v143 compiler
- Windows SDK
- CMake tools for Windows

2. **Install CUDA Toolkit** (optional but recommended)

- Download from [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads)
- Add CUDA to PATH during installation

3. **Install Microsoft MPI** (optional)

```powershell
# Download and install both:
# - msmpisetup.exe (runtime)
# - msmpisdk.msi (SDK)
```

4. **Install Ninja** (recommended)

```powershell
# Using Chocolatey
choco install ninja

# Or download from https://github.com/ninja-build/ninja/releases
```

## Build Steps

### **Option 1:** Using CMake Presets (Recommended)

```powershell
# Clone repository
git clone https://github.com/erenalyoruk/matgen.git
cd matgen

# Configure with MSVC
cmake --preset windows-msvc-release

# Build
cmake --build out/build/windows-msvc-release

# Run tests (optional)
ctest --preset windows-msvc-release
```

### **Option 2:** Manual Configuration

```powershell
# Open "x64 Native Tools Command Prompt for VS 2022"
mkdir build && cd build

cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
ninja
ctest
```

## Usage

```c
#include "matgen/algorithms/scaling.h"
#include "matgen/io/mtx_reader.h"
#include "matgen/io/mtx_writer.h"
#include "matgen/core/execution/policy.h"

// Read source matrix
matgen_csr_matrix_t* source = NULL;
matgen_mtx_read("input.mtx", &source, MATGEN_EXEC_AUTO);

// Scale 4x with nearest neighbor (CUDA if available)
matgen_csr_matrix_t* scaled = NULL;
matgen_scale_nearest_neighbor(source,
                               source->rows * 4,
                               source->cols * 4,
                               MATGEN_COLLISION_SUM,
                               MATGEN_EXEC_AUTO,
                               &scaled);

// Write result
matgen_mtx_write("output.mtx", scaled);

// Cleanup
matgen_csr_destroy(source);
matgen_csr_destroy(scaled);
```
