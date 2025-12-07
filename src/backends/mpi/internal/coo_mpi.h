#ifndef MATGEN_BACKENDS_MPI_INTERNAL_COO_MPI_H
#define MATGEN_BACKENDS_MPI_INTERNAL_COO_MPI_H

/**
 * @file coo_mpi.h
 * @brief Internal header for MPI-distributed COO matrix operations
 *
 * This is an internal header used only by the library implementation.
 * Users should use the public API in <matgen/core/matrix/coo.h> instead.
 *
 * MPI Strategy:
 *   - Each rank owns a subset of triplets (row, col, value)
 *   - Global operations require communication (sort, merge)
 *   - Local operations can be performed independently
 */

#include <mpi.h>

#include "matgen/core/matrix/coo.h"
#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// MPI COO Operations
// =============================================================================

/**
 * @brief Create a new COO matrix (MPI-distributed)
 *
 * Creates a local portion of the matrix on each rank.
 * The global dimensions are the same on all ranks.
 *
 * @param rows Global number of rows
 * @param cols Global number of columns
 * @param nnz_hint Expected number of local non-zeros on this rank
 * @return Pointer to local matrix portion, or NULL on error
 */
matgen_coo_matrix_t* matgen_coo_create_mpi(matgen_index_t rows,
                                           matgen_index_t cols,
                                           matgen_size_t nnz_hint);

/**
 * @brief Sort COO matrix entries globally by (row, col) order (MPI)
 *
 * Uses parallel sample sort algorithm:
 *   1. Local sort on each rank
 *   2. Sample and broadcast pivots
 *   3. Redistribute based on pivots
 *   4. Final local sort
 *
 * @param matrix Matrix to sort
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_coo_sort_mpi(matgen_coo_matrix_t* matrix);

/**
 * @brief Sum duplicate entries in a sorted COO matrix (MPI)
 *
 * Assumes matrix is already sorted globally. Each rank processes local
 * duplicates, then handles boundary duplicates with neighboring ranks.
 *
 * @param matrix Matrix to process (must be globally sorted)
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_coo_sum_duplicates_mpi(matgen_coo_matrix_t* matrix);

/**
 * @brief Merge duplicate entries using collision policy (MPI)
 *
 * @param matrix Matrix to process (must be globally sorted)
 * @param policy Collision policy (SUM, AVG, MAX, MIN, LAST)
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_coo_merge_duplicates_mpi(
    matgen_coo_matrix_t* matrix, matgen_collision_policy_t policy);

/**
 * @brief Get the total global NNZ across all ranks
 *
 * @param matrix Local matrix portion
 * @param global_nnz Output: total NNZ across all ranks
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_coo_get_global_nnz(const matgen_coo_matrix_t* matrix,
                                         matgen_size_t* global_nnz);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_BACKENDS_MPI_INTERNAL_COO_MPI_H
