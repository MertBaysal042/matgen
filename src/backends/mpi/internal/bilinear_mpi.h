#ifndef MATGEN_BACKENDS_MPI_INTERNAL_BILINEAR_MPI_H
#define MATGEN_BACKENDS_MPI_INTERNAL_BILINEAR_MPI_H

/**
 * @file bilinear_mpi.h
 * @brief Internal header for MPI-distributed bilinear interpolation
 *
 * This is an internal header used only by the library implementation.
 * Users should use the public API in <matgen/algorithms/scaling.h> instead.
 *
 * MPI Strategy for Bilinear Scaling:
 *   1. Each rank processes its assigned source rows independently
 *   2. Generate local triplets (row, col, value)
 *   3. Redistribute triplets based on destination row ownership
 *   4. Each rank builds its CSR portion from received triplets
 *   5. Sort and deduplicate locally
 */

#include <mpi.h>

#include "matgen/core/matrix/csr.h"
#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Scale sparse matrix using bilinear interpolation (MPI)
 *
 * Distributed algorithm:
 *   - Source matrix is distributed by rows across ranks
 *   - Each rank scales its portion independently
 *   - All-to-all communication redistributes entries by destination row
 *   - Each rank builds final CSR for its destination rows
 *
 * @param source Source matrix (local portion, CSR format)
 * @param new_rows Target global number of rows
 * @param new_cols Target global number of columns
 * @param result Output: scaled matrix local portion (CSR format)
 * @return MATGEN_SUCCESS on success, error code otherwise
 */
matgen_error_t matgen_scale_bilinear_mpi(const matgen_csr_matrix_t* source,
                                         matgen_index_t new_rows,
                                         matgen_index_t new_cols,
                                         matgen_csr_matrix_t** result);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_BACKENDS_MPI_INTERNAL_BILINEAR_MPI_H
