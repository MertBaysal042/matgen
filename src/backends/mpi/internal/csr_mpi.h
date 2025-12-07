#ifndef MATGEN_BACKENDS_MPI_INTERNAL_CSR_MPI_H
#define MATGEN_BACKENDS_MPI_INTERNAL_CSR_MPI_H

/**
 * @file csr_mpi.h
 * @brief Internal header for MPI-distributed CSR matrix operations
 *
 * This is an internal header used only by the library implementation.
 * Users should use the public API in <matgen/core/matrix/csr.h> instead.
 *
 * MPI Distribution Strategy:
 *   - Row-wise partitioning: each rank owns a contiguous range of rows
 *   - Global dimensions known to all ranks
 *   - Each rank stores: local row_ptr, col_indices, values
 *   - row_ptr is relative to local storage (starts at 0)
 */

#include <mpi.h>

#include "matgen/core/matrix/csr.h"
#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Distribution information for MPI CSR matrix
 */
typedef struct {
  matgen_index_t global_rows;  // Total rows across all ranks
  matgen_index_t
      local_row_start;  // First row owned by this rank (global index)
  matgen_index_t local_row_count;  // Number of rows owned by this rank
  int rank;                        // MPI rank
  int size;                        // Total number of MPI processes
} matgen_csr_mpi_dist_t;

// =============================================================================
// MPI CSR Operations
// =============================================================================

/**
 * @brief Create a new CSR matrix (MPI-distributed, row-wise)
 *
 * Creates a local portion of the matrix on each rank using row-wise
 * partitioning. Each rank owns local_row_count rows.
 *
 * @param rows Global number of rows
 * @param cols Global number of columns
 * @param nnz Local number of non-zeros on this rank
 * @return Pointer to local CSR matrix, or NULL on error
 */
matgen_csr_matrix_t* matgen_csr_create_mpi(matgen_index_t rows,
                                           matgen_index_t cols,
                                           matgen_size_t nnz);

/**
 * @brief Get distribution information for this rank
 *
 * Computes which rows are owned by this rank based on uniform distribution.
 *
 * @param global_rows Total number of rows
 * @param dist Output: distribution information
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_csr_get_distribution(matgen_index_t global_rows,
                                           matgen_csr_mpi_dist_t* dist);

/**
 * @brief Get the total global NNZ across all ranks
 *
 * @param local_nnz Local number of non-zeros
 * @param global_nnz Output: total NNZ across all ranks
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_csr_get_global_nnz(matgen_size_t local_nnz,
                                         matgen_size_t* global_nnz);

/**
 * @brief Gather distributed CSR matrix to rank 0
 *
 * Collects all local portions to rank 0, creating a complete global matrix.
 * Other ranks receive NULL.
 *
 * @param local_matrix Local portion of the matrix
 * @param global_rows Total number of rows
 * @param global_cols Total number of columns
 * @return Global matrix on rank 0, NULL on other ranks, or NULL on error
 */
matgen_csr_matrix_t* matgen_csr_gather(const matgen_csr_matrix_t* local_matrix,
                                       matgen_index_t global_rows,
                                       matgen_index_t global_cols);

/**
 * @brief Broadcast CSR matrix from rank 0 to all ranks
 *
 * Distributes a matrix from rank 0 to all other ranks using row-wise
 * partitioning. On rank 0, input is the full matrix; on other ranks, input
 * should be NULL.
 *
 * @param global_matrix Full matrix on rank 0, NULL on other ranks
 * @return Local portion on all ranks, or NULL on error
 */
matgen_csr_matrix_t* matgen_csr_scatter(
    const matgen_csr_matrix_t* global_matrix);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_BACKENDS_MPI_INTERNAL_CSR_MPI_H
