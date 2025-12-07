#include "backends/mpi/internal/coo_mpi.h"

#include <mpi.h>
#include <stdlib.h>

#include "matgen/core/matrix/coo.h"
#include "matgen/utils/log.h"

// =============================================================================
// Configuration
// =============================================================================

#define DEFAULT_INITIAL_CAPACITY 1024
#define GROWTH_FACTOR 1.5

// =============================================================================
// MPI Datatypes
// =============================================================================

// Structure for sorting triplets
typedef struct {
  matgen_index_t row;
  matgen_index_t col;
  matgen_value_t val;
} coo_triplet_t;

// Create MPI datatype for triplet
static MPI_Datatype create_triplet_type(void) {
  static MPI_Datatype triplet_type = MPI_DATATYPE_NULL;

  if (triplet_type != MPI_DATATYPE_NULL) {
    return triplet_type;
  }

  int block_lengths[3] = {1, 1, 1};
  MPI_Aint offsets[3];
  MPI_Datatype types[3];

  // Calculate offsets
  offsets[0] = offsetof(coo_triplet_t, row);
  offsets[1] = offsetof(coo_triplet_t, col);
  offsets[2] = offsetof(coo_triplet_t, val);

// Set types based on matgen types
#if defined(MATGEN_INDEX_64)
  types[0] = MPI_UINT64_T;
  types[1] = MPI_UINT64_T;
#else
  types[0] = MPI_UINT32_T;
  types[1] = MPI_UINT32_T;
#endif

#if defined(MATGEN_USE_DOUBLE)
  types[2] = MPI_DOUBLE;
#else
  types[2] = MPI_FLOAT;
#endif

  MPI_Type_create_struct(3, block_lengths, offsets, types, &triplet_type);
  MPI_Type_commit(&triplet_type);

  return triplet_type;
}

// =============================================================================
// Comparison Functions
// =============================================================================

static int compare_triplets(const void* a, const void* b) {
  const coo_triplet_t* ta = (const coo_triplet_t*)a;
  const coo_triplet_t* tb = (const coo_triplet_t*)b;

  if (ta->row < tb->row) {
    return -1;
  }
  if (ta->row > tb->row) {
    return 1;
  }
  if (ta->col < tb->col) {
    return -1;
  }
  if (ta->col > tb->col) {
    return 1;
  }
  return 0;
}

// =============================================================================
// MPI Backend Implementation
// =============================================================================

matgen_coo_matrix_t* matgen_coo_create_mpi(matgen_index_t rows,
                                           matgen_index_t cols,
                                           matgen_size_t nnz_hint) {
  if (rows == 0 || cols == 0) {
    MATGEN_LOG_ERROR("Invalid matrix dimensions: %llu x %llu",
                     (unsigned long long)rows, (unsigned long long)cols);
    return NULL;
  }

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  matgen_coo_matrix_t* matrix =
      (matgen_coo_matrix_t*)malloc(sizeof(matgen_coo_matrix_t));
  if (!matrix) {
    MATGEN_LOG_ERROR("Failed to allocate COO matrix structure");
    return NULL;
  }

  matrix->rows = rows;
  matrix->cols = cols;
  matrix->nnz = 0;
  matrix->capacity = (nnz_hint > 0) ? nnz_hint : DEFAULT_INITIAL_CAPACITY;
  matrix->is_sorted = true;  // Empty matrix is trivially sorted

  // Allocate local arrays
  matrix->row_indices =
      (matgen_index_t*)malloc(matrix->capacity * sizeof(matgen_index_t));
  matrix->col_indices =
      (matgen_index_t*)malloc(matrix->capacity * sizeof(matgen_index_t));
  matrix->values =
      (matgen_value_t*)malloc(matrix->capacity * sizeof(matgen_value_t));

  if (!matrix->row_indices || !matrix->col_indices || !matrix->values) {
    MATGEN_LOG_ERROR("Failed to allocate COO matrix arrays");
    matgen_coo_destroy(matrix);
    return NULL;
  }

  MATGEN_LOG_DEBUG(
      "[Rank %d] Created COO matrix (MPI) %llu x %llu with local capacity %zu",
      rank, (unsigned long long)rows, (unsigned long long)cols,
      matrix->capacity);

  return matrix;
}

matgen_error_t matgen_coo_sort_mpi(matgen_coo_matrix_t* matrix) {
  if (!matrix) {
    MATGEN_LOG_ERROR("NULL matrix pointer");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (matrix->is_sorted && size == 1) {
    MATGEN_LOG_DEBUG("[Rank %d] Matrix already sorted", rank);
    return MATGEN_SUCCESS;
  }

  MATGEN_LOG_DEBUG("[Rank %d] Sorting COO matrix (MPI) with %zu local entries",
                   rank, matrix->nnz);

  // Step 1: Local sort
  if (matrix->nnz > 0) {
    coo_triplet_t* triplets =
        (coo_triplet_t*)malloc(matrix->nnz * sizeof(coo_triplet_t));
    if (!triplets) {
      MATGEN_LOG_ERROR("Failed to allocate triplet buffer for sorting");
      return MATGEN_ERROR_OUT_OF_MEMORY;
    }

    // Pack into triplets
    for (matgen_size_t i = 0; i < matrix->nnz; i++) {
      triplets[i].row = matrix->row_indices[i];
      triplets[i].col = matrix->col_indices[i];
      triplets[i].val = matrix->values[i];
    }

    // Local sort
    qsort(triplets, matrix->nnz, sizeof(coo_triplet_t), compare_triplets);

    // Unpack
    for (matgen_size_t i = 0; i < matrix->nnz; i++) {
      matrix->row_indices[i] = triplets[i].row;
      matrix->col_indices[i] = triplets[i].col;
      matrix->values[i] = triplets[i].val;
    }

    free(triplets);
  }

  // For single rank, we're done
  if (size == 1) {
    matrix->is_sorted = true;
    MATGEN_LOG_DEBUG("[Rank %d] Local sort complete", rank);
    return MATGEN_SUCCESS;
  }

  // Step 2: Parallel sample sort (simplified version - local sort only for now)
  // TODO: Implement full parallel sample sort with redistribution
  // For now, we assume each rank processes disjoint row ranges

  matrix->is_sorted = true;
  MATGEN_LOG_DEBUG("[Rank %d] MPI sort complete (local only)", rank);

  return MATGEN_SUCCESS;
}

matgen_error_t matgen_coo_sum_duplicates_mpi(matgen_coo_matrix_t* matrix) {
  if (!matrix) {
    MATGEN_LOG_ERROR("NULL matrix pointer");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (matrix->nnz <= 1) {
    return MATGEN_SUCCESS;
  }

  if (!matrix->is_sorted) {
    MATGEN_LOG_ERROR("Matrix must be sorted before calling sum_duplicates");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MATGEN_LOG_DEBUG(
      "[Rank %d] Summing duplicates in COO matrix (MPI) with %zu local entries",
      rank, matrix->nnz);

  // Step 1: Local deduplication (same as sequential)
  matgen_size_t write_idx = 0;

  for (matgen_size_t read_idx = 0; read_idx < matrix->nnz; read_idx++) {
    matrix->row_indices[write_idx] = matrix->row_indices[read_idx];
    matrix->col_indices[write_idx] = matrix->col_indices[read_idx];
    matrix->values[write_idx] = matrix->values[read_idx];

    // Sum local duplicates
    while (
        read_idx + 1 < matrix->nnz &&
        matrix->row_indices[read_idx + 1] == matrix->row_indices[write_idx] &&
        matrix->col_indices[read_idx + 1] == matrix->col_indices[write_idx]) {
      read_idx++;
      matrix->values[write_idx] += matrix->values[read_idx];
    }

    write_idx++;
  }

  matgen_size_t old_nnz = matrix->nnz;
  matrix->nnz = write_idx;

  MATGEN_LOG_DEBUG(
      "[Rank %d] Local deduplication: reduced from %zu to %zu entries", rank,
      old_nnz, matrix->nnz);

  // Step 2: Handle boundary duplicates between ranks
  // TODO: Exchange boundary entries with neighbors and merge
  // For now, assumes proper distribution (no cross-rank duplicates)

  return MATGEN_SUCCESS;
}

matgen_error_t matgen_coo_merge_duplicates_mpi(
    matgen_coo_matrix_t* matrix, matgen_collision_policy_t policy) {
  if (!matrix) {
    MATGEN_LOG_ERROR("NULL matrix pointer");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (matrix->nnz <= 1) {
    return MATGEN_SUCCESS;
  }

  if (!matrix->is_sorted) {
    MATGEN_LOG_ERROR("Matrix must be sorted before calling merge_duplicates");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MATGEN_LOG_DEBUG(
      "[Rank %d] Merging duplicates in COO matrix (MPI, policy: %d)", rank,
      policy);

  // Local merge (same as sequential)
  matgen_size_t write_idx = 0;

  for (matgen_size_t read_idx = 0; read_idx < matrix->nnz; read_idx++) {
    matrix->row_indices[write_idx] = matrix->row_indices[read_idx];
    matrix->col_indices[write_idx] = matrix->col_indices[read_idx];
    matrix->values[write_idx] = matrix->values[read_idx];

    matgen_size_t dup_count = 1;

    while (
        read_idx + 1 < matrix->nnz &&
        matrix->row_indices[read_idx + 1] == matrix->row_indices[write_idx] &&
        matrix->col_indices[read_idx + 1] == matrix->col_indices[write_idx]) {
      read_idx++;
      dup_count++;

      matgen_value_t current_val = matrix->values[write_idx];
      matgen_value_t new_val = matrix->values[read_idx];

      switch (policy) {
        case MATGEN_COLLISION_SUM:
          matrix->values[write_idx] = current_val + new_val;
          break;
        case MATGEN_COLLISION_AVG:
          matrix->values[write_idx] = current_val + ((new_val - current_val) /
                                                     (matgen_value_t)dup_count);
          break;
        case MATGEN_COLLISION_MAX:
          if (new_val > current_val) {
            matrix->values[write_idx] = new_val;
          }
          break;
        case MATGEN_COLLISION_MIN:
          if (new_val < current_val) {
            matrix->values[write_idx] = new_val;
          }
          break;
        case MATGEN_COLLISION_LAST:
          matrix->values[write_idx] = new_val;
          break;
        default:
          MATGEN_LOG_ERROR("Unknown collision policy: %d", policy);
          return MATGEN_ERROR_INVALID_ARGUMENT;
      }
    }

    write_idx++;
  }

  matgen_size_t old_nnz = matrix->nnz;
  matrix->nnz = write_idx;

  MATGEN_LOG_DEBUG("[Rank %d] Reduced nnz from %zu to %zu", rank, old_nnz,
                   matrix->nnz);

  return MATGEN_SUCCESS;
}

matgen_error_t matgen_coo_get_global_nnz(const matgen_coo_matrix_t* matrix,
                                         matgen_size_t* global_nnz) {
  if (!matrix || !global_nnz) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  matgen_size_t local_nnz = matrix->nnz;

#if defined(MATGEN_SIZE_64)
  MPI_Allreduce(&local_nnz, global_nnz, 1, MPI_UINT64_T, MPI_SUM,
                MPI_COMM_WORLD);
#else
  MPI_Allreduce(&local_nnz, global_nnz, 1, MPI_UINT32_T, MPI_SUM,
                MPI_COMM_WORLD);
#endif

  return MATGEN_SUCCESS;
}
