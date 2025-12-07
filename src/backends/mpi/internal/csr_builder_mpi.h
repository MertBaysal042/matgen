#ifndef MATGEN_BACKENDS_MPI_INTERNAL_CSR_BUILDER_MPI_H
#define MATGEN_BACKENDS_MPI_INTERNAL_CSR_BUILDER_MPI_H

/**
 * @file csr_builder_mpi.h
 * @brief Internal header for MPI-distributed CSR builder
 *
 * MPI Builder Strategy:
 *   - Each rank builds its local portion independently
 *   - Row distribution is implicit (based on global row index % size)
 *   - Entries are added locally (no communication during build)
 *   - Finalization creates local CSR matrix portion
 */

#include <mpi.h>

#include "matgen/core/matrix/csr_builder.h"

#ifdef __cplusplus
extern "C" {
#endif

// MPI implementation functions
matgen_csr_builder_t* matgen_csr_builder_create_mpi(matgen_index_t rows,
                                                    matgen_index_t cols,
                                                    matgen_size_t est_nnz);

void matgen_csr_builder_destroy_mpi(matgen_csr_builder_t* builder);

matgen_error_t matgen_csr_builder_add_mpi(matgen_csr_builder_t* builder,
                                          matgen_index_t row,
                                          matgen_index_t col,
                                          matgen_value_t value);

matgen_size_t matgen_csr_builder_get_nnz_mpi(
    const matgen_csr_builder_t* builder);

matgen_csr_matrix_t* matgen_csr_builder_finalize_mpi(
    matgen_csr_builder_t* builder);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_BACKENDS_MPI_INTERNAL_CSR_BUILDER_MPI_H
