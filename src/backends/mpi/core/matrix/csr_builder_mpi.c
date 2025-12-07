#include "backends/mpi/internal/csr_builder_mpi.h"

#include <mpi.h>
#include <stdlib.h>

#include "backends/mpi/internal/csr_mpi.h"
#include "core/matrix/csr_builder_internal.h"
#include "matgen/utils/log.h"

// =============================================================================
// Builder Creation and Destruction
// =============================================================================

matgen_csr_builder_t* matgen_csr_builder_create_mpi(matgen_index_t rows,
                                                    matgen_index_t cols,
                                                    matgen_size_t est_nnz) {
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Get distribution info
  matgen_csr_mpi_dist_t dist;
  if (matgen_csr_get_distribution(rows, &dist) != MATGEN_SUCCESS) {
    MATGEN_LOG_ERROR("Failed to compute distribution");
    return NULL;
  }

  matgen_csr_builder_t* builder =
      (matgen_csr_builder_t*)malloc(sizeof(matgen_csr_builder_t));
  if (!builder) {
    return NULL;
  }

  builder->rows = rows;  // Store global row count
  builder->cols = cols;
  builder->est_nnz = est_nnz;
  builder->policy = MATGEN_EXEC_MPI;
  builder->collision_policy = MATGEN_COLLISION_SUM;
  builder->finalized = false;
  builder->backend.seq.entry_count = 0;  // Reuse seq structure for local data

  // Allocate row buffers only for local rows
  builder->backend.seq.row_buffers = (csr_row_buffer_t*)malloc(
      dist.local_row_count * sizeof(csr_row_buffer_t));
  if (!builder->backend.seq.row_buffers) {
    free(builder);
    return NULL;
  }

  // Initialize all local row buffers
  for (matgen_index_t r = 0; r < dist.local_row_count; r++) {
    csr_builder_init_row_buffer(&builder->backend.seq.row_buffers[r]);
  }

  MATGEN_LOG_DEBUG(
      "[Rank %d] Created CSR builder (MPI): global %llu x %llu, local rows "
      "[%llu, %llu), est_nnz=%zu",
      rank, (unsigned long long)rows, (unsigned long long)cols,
      (unsigned long long)dist.local_row_start,
      (unsigned long long)(dist.local_row_start + dist.local_row_count),
      est_nnz);

  return builder;
}

void matgen_csr_builder_destroy_mpi(matgen_csr_builder_t* builder) {
  if (!builder) {
    return;
  }

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (builder->backend.seq.row_buffers) {
    // Get local row count
    matgen_csr_mpi_dist_t dist;
    if (matgen_csr_get_distribution(builder->rows, &dist) == MATGEN_SUCCESS) {
      for (matgen_index_t r = 0; r < dist.local_row_count; r++) {
        csr_builder_destroy_row_buffer(&builder->backend.seq.row_buffers[r]);
      }
    }
    free(builder->backend.seq.row_buffers);
  }

  free(builder);
}

// =============================================================================
// Entry Addition
// =============================================================================

matgen_error_t matgen_csr_builder_add_mpi(matgen_csr_builder_t* builder,
                                          matgen_index_t row,
                                          matgen_index_t col,
                                          matgen_value_t value) {
  if (!builder || builder->finalized) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (row >= builder->rows || col >= builder->cols) {
    MATGEN_LOG_ERROR("Index out of bounds: (%llu, %llu) in %llu x %llu matrix",
                     (unsigned long long)row, (unsigned long long)col,
                     (unsigned long long)builder->rows,
                     (unsigned long long)builder->cols);
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  // Get distribution info
  matgen_csr_mpi_dist_t dist;
  if (matgen_csr_get_distribution(builder->rows, &dist) != MATGEN_SUCCESS) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  // Check if this row belongs to this rank
  if (row < dist.local_row_start ||
      row >= dist.local_row_start + dist.local_row_count) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MATGEN_LOG_ERROR(
        "[Rank %d] Row %llu not in local range [%llu, %llu). Entry ignored.",
        rank, (unsigned long long)row, (unsigned long long)dist.local_row_start,
        (unsigned long long)(dist.local_row_start + dist.local_row_count));
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  // Convert global row to local row index
  matgen_index_t local_row = row - dist.local_row_start;

  // Add to local row buffer
  matgen_error_t err = csr_builder_add_to_row_buffer(
      &builder->backend.seq.row_buffers[local_row], col, value);
  if (err == MATGEN_SUCCESS) {
    builder->backend.seq.entry_count++;
  }

  return err;
}

// =============================================================================
// Query Functions
// =============================================================================

matgen_size_t matgen_csr_builder_get_nnz_mpi(
    const matgen_csr_builder_t* builder) {
  if (!builder) {
    return 0;
  }

  // Return local entry count
  matgen_size_t local_nnz = builder->backend.seq.entry_count;

  // Optionally, could return global NNZ with MPI_Allreduce
  // For now, return local count
  return local_nnz;
}

// =============================================================================
// Finalization
// =============================================================================

matgen_csr_matrix_t* matgen_csr_builder_finalize_mpi(
    matgen_csr_builder_t* builder) {
  if (!builder || builder->finalized) {
    return NULL;
  }

  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MATGEN_LOG_DEBUG("[Rank %d] Finalizing CSR builder (MPI)", rank);

  builder->finalized = true;

  // Get distribution info
  matgen_csr_mpi_dist_t dist;
  if (matgen_csr_get_distribution(builder->rows, &dist) != MATGEN_SUCCESS) {
    MATGEN_LOG_ERROR("Failed to compute distribution");
    return NULL;
  }

  matgen_index_t local_row_count = dist.local_row_count;

  // Phase 1: Count entries per local row
  matgen_size_t total_local_nnz = 0;
  matgen_size_t* row_counts =
      (matgen_size_t*)malloc(local_row_count * sizeof(matgen_size_t));
  if (!row_counts) {
    return NULL;
  }

  for (matgen_index_t r = 0; r < local_row_count; r++) {
    csr_row_buffer_t* row_buf = &builder->backend.seq.row_buffers[r];
    matgen_size_t count = 0;

    // Count hash table entries
    for (int i = 0; i < MATGEN_CSR_BUILDER_HASH_SIZE; i++) {
      if (row_buf->hash_table[i].col != (matgen_index_t)-1) {
        count++;
      }
    }

    // Add overflow entries
    count += row_buf->overflow_count;

    row_counts[r] = count;
    total_local_nnz += count;
  }

  MATGEN_LOG_DEBUG(
      "[Rank %d] Total local unique entries: %zu (input entries: %zu)", rank,
      total_local_nnz, builder->backend.seq.entry_count);

  // Phase 2: Create local CSR matrix and compute row_ptr
  matgen_csr_matrix_t* csr =
      matgen_csr_create_mpi(builder->rows, builder->cols, total_local_nnz);
  if (!csr) {
    free(row_counts);
    return NULL;
  }

  csr->row_ptr[0] = 0;
  for (matgen_index_t r = 0; r < local_row_count; r++) {
    csr->row_ptr[r + 1] = csr->row_ptr[r] + row_counts[r];
  }

  free(row_counts);

  // Phase 3: Extract and sort entries for each local row
  for (matgen_index_t r = 0; r < local_row_count; r++) {
    csr_row_buffer_t* row_buf = &builder->backend.seq.row_buffers[r];

    matgen_size_t row_start = csr->row_ptr[r];
    matgen_size_t row_nnz = csr->row_ptr[r + 1] - row_start;

    if (row_nnz == 0) {
      continue;
    }

    // Collect entries from hash table and overflow
    csr_hash_entry_t* entries =
        (csr_hash_entry_t*)malloc(row_nnz * sizeof(csr_hash_entry_t));
    if (!entries) {
      matgen_csr_destroy(csr);
      matgen_csr_builder_destroy_mpi(builder);
      return NULL;
    }

    matgen_size_t idx = 0;

    // Collect from hash table
    for (int i = 0; i < MATGEN_CSR_BUILDER_HASH_SIZE; i++) {
      if (row_buf->hash_table[i].col != (matgen_index_t)-1) {
        entries[idx++] = row_buf->hash_table[i];
      }
    }

    // Collect from overflow
    for (matgen_size_t i = 0; i < row_buf->overflow_count; i++) {
      entries[idx++] = row_buf->overflow[i];
    }

    // Sort by column
    qsort(entries, row_nnz, sizeof(csr_hash_entry_t),
          csr_builder_compare_entries);

    // Write to CSR
    for (matgen_size_t i = 0; i < row_nnz; i++) {
      csr->col_indices[row_start + i] = entries[i].col;
      csr->values[row_start + i] = entries[i].val;
    }

    free(entries);
  }

  // Get global statistics
  matgen_size_t global_nnz = 0;
#if defined(MATGEN_SIZE_64)
  MPI_Allreduce(&total_local_nnz, &global_nnz, 1, MPI_UINT64_T, MPI_SUM,
                MPI_COMM_WORLD);
#else
  MPI_Allreduce(&total_local_nnz, &global_nnz, 1, MPI_UINT32_T, MPI_SUM,
                MPI_COMM_WORLD);
#endif

  // Cleanup builder
  matgen_csr_builder_destroy_mpi(builder);

  MATGEN_LOG_DEBUG(
      "[Rank %d] CSR builder finalized: global %llu x %llu, local nnz=%zu, "
      "global nnz=%zu",
      rank, (unsigned long long)csr->rows, (unsigned long long)csr->cols,
      total_local_nnz, global_nnz);

  return csr;
}
