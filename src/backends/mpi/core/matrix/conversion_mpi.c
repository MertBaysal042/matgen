#include "backends/mpi/internal/conversion_mpi.h"

#include <mpi.h>
#include <string.h>

#include "backends/mpi/internal/coo_mpi.h"
#include "backends/mpi/internal/csr_mpi.h"
#include "matgen/core/matrix/coo.h"
#include "matgen/core/matrix/csr.h"
#include "matgen/utils/log.h"

// =============================================================================
// COO to CSR Conversion (MPI)
// =============================================================================

matgen_csr_matrix_t* matgen_coo_to_csr_mpi(const matgen_coo_matrix_t* coo) {
  if (!coo) {
    MATGEN_LOG_ERROR("NULL COO matrix pointer");
    return NULL;
  }

  if (!matgen_coo_validate(coo)) {
    MATGEN_LOG_ERROR("Invalid COO matrix");
    return NULL;
  }

  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MATGEN_LOG_DEBUG(
      "[Rank %d] Converting local COO (%llu x %llu, local_nnz=%zu) to CSR "
      "(MPI)",
      rank, (unsigned long long)coo->rows, (unsigned long long)coo->cols,
      coo->nnz);

  // Get distribution info to determine local row range
  matgen_csr_mpi_dist_t dist;
  if (matgen_csr_get_distribution(coo->rows, &dist) != MATGEN_SUCCESS) {
    MATGEN_LOG_ERROR("Failed to compute distribution");
    return NULL;
  }

  // Create local CSR matrix
  // Note: coo->rows here should actually be the GLOBAL row count
  // The CSR will be created with local row count based on distribution
  matgen_csr_matrix_t* csr =
      matgen_csr_create_mpi(coo->rows, coo->cols, coo->nnz);
  if (!csr) {
    return NULL;
  }

  // Handle empty local portion
  if (coo->nnz == 0) {
    MATGEN_LOG_DEBUG("[Rank %d] Empty local matrix, conversion trivial", rank);
    return csr;
  }

  // If COO is not sorted, we need to sort it first
  matgen_coo_matrix_t* coo_sorted = NULL;
  const matgen_coo_matrix_t* coo_to_use = coo;

  if (!coo->is_sorted) {
    MATGEN_LOG_DEBUG("[Rank %d] COO matrix not sorted, creating sorted copy",
                     rank);
    coo_sorted = matgen_coo_create_mpi(coo->rows, coo->cols, coo->nnz);
    if (!coo_sorted) {
      matgen_csr_destroy(csr);
      return NULL;
    }

    // Copy data
    memcpy(coo_sorted->row_indices, coo->row_indices,
           coo->nnz * sizeof(matgen_index_t));
    memcpy(coo_sorted->col_indices, coo->col_indices,
           coo->nnz * sizeof(matgen_index_t));
    memcpy(coo_sorted->values, coo->values, coo->nnz * sizeof(matgen_value_t));
    coo_sorted->nnz = coo->nnz;
    coo_sorted->is_sorted = false;

    // Sort the copy using MPI backend
    if (matgen_coo_sort_mpi(coo_sorted) != MATGEN_SUCCESS) {
      MATGEN_LOG_ERROR("Failed to sort COO matrix");
      matgen_coo_destroy(coo_sorted);
      matgen_csr_destroy(csr);
      return NULL;
    }

    coo_to_use = coo_sorted;
  }

  // Convert COO entries to local CSR format
  // Note: COO row indices are global, need to convert to local
  matgen_index_t local_row_start = dist.local_row_start;
  matgen_index_t local_row_count = dist.local_row_count;

  // Count non-zeros per local row
  for (matgen_size_t i = 0; i < coo_to_use->nnz; i++) {
    matgen_index_t global_row = coo_to_use->row_indices[i];

    // Convert to local row index
    if (global_row >= local_row_start &&
        global_row < local_row_start + local_row_count) {
      matgen_index_t local_row = global_row - local_row_start;
      csr->row_ptr[local_row + 1]++;
    } else {
      MATGEN_LOG_ERROR(
          "[Rank %d] COO entry with row %llu not in local range [%llu, %llu)",
          rank, (unsigned long long)global_row,
          (unsigned long long)local_row_start,
          (unsigned long long)(local_row_start + local_row_count));
      if (coo_sorted) {
        matgen_coo_destroy(coo_sorted);
      }
      matgen_csr_destroy(csr);
      return NULL;
    }
  }

  // Convert counts to cumulative sum (prefix sum)
  for (matgen_index_t i = 0; i < local_row_count; i++) {
    csr->row_ptr[i + 1] += csr->row_ptr[i];
  }

  // Fill in column indices and values
  for (matgen_size_t i = 0; i < coo_to_use->nnz; i++) {
    matgen_index_t global_row = coo_to_use->row_indices[i];
    matgen_index_t local_row = global_row - local_row_start;

    matgen_size_t dest = csr->row_ptr[local_row];

    csr->col_indices[dest] = coo_to_use->col_indices[i];
    csr->values[dest] = coo_to_use->values[i];

    csr->row_ptr[local_row]++;
  }

  // Restore row_ptr (shift back)
  for (matgen_index_t i = local_row_count; i > 0; i--) {
    csr->row_ptr[i] = csr->row_ptr[i - 1];
  }
  csr->row_ptr[0] = 0;

  // Clean up sorted copy if we made one
  if (coo_sorted) {
    matgen_coo_destroy(coo_sorted);
  }

  MATGEN_LOG_DEBUG("[Rank %d] COO to CSR conversion complete (MPI)", rank);

  return csr;
}

// =============================================================================
// CSR to COO Conversion (MPI)
// =============================================================================

matgen_coo_matrix_t* matgen_csr_to_coo_mpi(const matgen_csr_matrix_t* csr) {
  if (!csr) {
    MATGEN_LOG_ERROR("NULL CSR matrix pointer");
    return NULL;
  }

  if (!matgen_csr_validate(csr)) {
    MATGEN_LOG_ERROR("Invalid CSR matrix");
    return NULL;
  }

  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MATGEN_LOG_DEBUG(
      "[Rank %d] Converting local CSR (%llu x %llu, local_nnz=%zu) to COO "
      "(MPI)",
      rank, (unsigned long long)csr->rows, (unsigned long long)csr->cols,
      csr->nnz);

  // Get distribution info to compute global row indices
  matgen_csr_mpi_dist_t dist;
  // Note: We need to know the global row count - this is a challenge
  // For now, we'll compute it from local row counts
  matgen_index_t local_rows = csr->rows;
  matgen_index_t global_rows = 0;

#if defined(MATGEN_INDEX_64)
  MPI_Allreduce(&local_rows, &global_rows, 1, MPI_UINT64_T, MPI_SUM,
                MPI_COMM_WORLD);
#else
  MPI_Allreduce(&local_rows, &global_rows, 1, MPI_UINT32_T, MPI_SUM,
                MPI_COMM_WORLD);
#endif

  if (matgen_csr_get_distribution(global_rows, &dist) != MATGEN_SUCCESS) {
    MATGEN_LOG_ERROR("Failed to compute distribution");
    return NULL;
  }

  // Create local COO matrix
  matgen_coo_matrix_t* coo =
      matgen_coo_create_mpi(global_rows, csr->cols, csr->nnz);
  if (!coo) {
    return NULL;
  }

  // Handle empty local portion
  if (csr->nnz == 0) {
    MATGEN_LOG_DEBUG("[Rank %d] Empty local matrix, conversion trivial", rank);
    return coo;
  }

  // Convert: iterate through each local row
  matgen_size_t coo_idx = 0;
  matgen_index_t local_row_start = dist.local_row_start;

  for (matgen_index_t local_row = 0; local_row < csr->rows; local_row++) {
    matgen_size_t row_start = csr->row_ptr[local_row];
    matgen_size_t row_end = csr->row_ptr[local_row + 1];

    // Convert local row index to global row index
    matgen_index_t global_row = local_row_start + local_row;

    for (matgen_size_t j = row_start; j < row_end; j++) {
      coo->row_indices[coo_idx] = global_row;
      coo->col_indices[coo_idx] = csr->col_indices[j];
      coo->values[coo_idx] = csr->values[j];
      coo_idx++;
    }
  }

  coo->nnz = csr->nnz;
  coo->is_sorted = true;  // CSR is always sorted

  MATGEN_LOG_DEBUG("[Rank %d] CSR to COO conversion complete (MPI)", rank);

  return coo;
}
