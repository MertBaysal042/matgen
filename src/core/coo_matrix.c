#include "matgen/core/coo_matrix.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matgen/util/log.h"

// =============================================================================
// Configuration
// =============================================================================

// Initial capacity if no hint provided
#define DEFAULT_INITIAL_CAPACITY 1024

// Growth factor when reallocating
#define GROWTH_FACTOR 1.5

// =============================================================================
// Creation and Destruction
// =============================================================================

matgen_coo_matrix_t* matgen_coo_create(size_t rows, size_t cols,
                                       size_t nnz_hint) {
  if (rows == 0 || cols == 0) {
    MATGEN_LOG_ERROR("Invalid matrix dimensions: %zu x %zu", rows, cols);
    return NULL;
  }

  matgen_coo_matrix_t* matrix =
      (matgen_coo_matrix_t*)malloc(sizeof(matgen_coo_matrix_t));
  if (!matrix) {
    MATGEN_LOG_ERROR("Failed to allocate COO matrix structure");
    return NULL;
  }

  matrix->rows = rows;
  matrix->cols = cols;
  matrix->nnz = 0;
  matrix->capacity = (nnz_hint) > 0 ? nnz_hint : DEFAULT_INITIAL_CAPACITY;
  matrix->is_sorted = true;  // Empty matrix is trivially sorted

  // Allocate arrays
  matrix->row_indices = (size_t*)malloc(matrix->capacity * sizeof(size_t));
  matrix->col_indices = (size_t*)malloc(matrix->capacity * sizeof(size_t));
  matrix->values = (double*)malloc(matrix->capacity * sizeof(double));

  if (!matrix->row_indices || !matrix->col_indices || !matrix->values) {
    MATGEN_LOG_ERROR("Failed to allocate COO matrix arrays");
    matgen_coo_destroy(matrix);
    return NULL;
  }

  MATGEN_LOG_DEBUG("Created COO matrix %zu x %zu with capacity %zu", rows, cols,
                   matrix->capacity);

  return matrix;
}

void matgen_coo_destroy(matgen_coo_matrix_t* matrix) {
  if (!matrix) {
    return;
  }

  MATGEN_LOG_DEBUG("Destroying COO matrix %zu x %zu (nnz: %zu)", matrix->rows,
                   matrix->cols, matrix->nnz);

  free(matrix->row_indices);
  free(matrix->col_indices);
  free(matrix->values);
  free(matrix);
}

// =============================================================================
// Internal Helper Functions
// =============================================================================

static int coo_resize(matgen_coo_matrix_t* matrix, size_t new_capacity) {
  MATGEN_LOG_DEBUG("Resizing COO matrix from capacity %zu to %zu",
                   matrix->capacity, new_capacity);

  size_t* new_rows =
      (size_t*)realloc(matrix->row_indices, new_capacity * sizeof(size_t));
  size_t* new_cols =
      (size_t*)realloc(matrix->col_indices, new_capacity * sizeof(size_t));
  double* new_vals =
      (double*)realloc(matrix->values, new_capacity * sizeof(double));

  if (!new_rows || !new_cols || !new_vals) {
    MATGEN_LOG_ERROR("Failed to resize COO matrix to capacity %zu",
                     new_capacity);
    // Realloc failed, original pointers are still valid
    return -1;
  }

  matrix->row_indices = new_rows;
  matrix->col_indices = new_cols;
  matrix->values = new_vals;
  matrix->capacity = new_capacity;

  return 0;
}

// =============================================================================
// Building the Matrix
// =============================================================================

int matgen_coo_add_entry(matgen_coo_matrix_t* matrix, size_t row, size_t col,
                         double value) {
  if (!matrix) {
    MATGEN_LOG_ERROR("NULL matrix pointer");
    return -1;
  }

  if (row >= matrix->rows || col >= matrix->cols) {
    MATGEN_LOG_ERROR("Index out of bounds: (%zu, %zu) for %zu x %zu matrix",
                     row, col, matrix->rows, matrix->cols);
    return -2;
  }

  // Grow array if needed
  if (matrix->nnz >= matrix->capacity) {
    size_t new_capacity =
        (size_t)((double)matrix->capacity * GROWTH_FACTOR) + 1;
    if (coo_resize(matrix, new_capacity) != 0) {
      return -3;
    }
  }

  // Add entry
  matrix->row_indices[matrix->nnz] = row;
  matrix->col_indices[matrix->nnz] = col;
  matrix->values[matrix->nnz] = value;
  matrix->nnz++;

  // Matrix is no longer sorted after adding
  matrix->is_sorted = false;

  MATGEN_LOG_TRACE("Added entry at (%zu, %zu) = %f, nnz now: %zu", row, col,
                   value, matrix->nnz);

  return 0;
}

// =============================================================================
// Sorting
// =============================================================================

// Comparison helper for sorting
static inline int compare_entry(size_t row1, size_t col1, size_t row2,
                                size_t col2) {
  if (row1 < row2) {
    return -1;
  }

  if (row1 > row2) {
    return 1;
  }

  if (col1 < col2) {
    return -1;
  }

  if (col1 > col2) {
    return 1;
  }

  return 0;
}

// Simple insertion sort - good for small or nearly-sorted arrays
static void insertion_sort_coo(matgen_coo_matrix_t* matrix) {
  for (size_t i = 1; i < matrix->nnz; i++) {
    size_t row_key = matrix->row_indices[i];
    size_t col_key = matrix->col_indices[i];
    double val_key = matrix->values[i];

    size_t j = i;
    while (j > 0 &&
           compare_entry(matrix->row_indices[j - 1], matrix->col_indices[j - 1],
                         row_key, col_key) > 0) {
      // Shift element right
      matrix->row_indices[j] = matrix->row_indices[j - 1];
      matrix->col_indices[j] = matrix->col_indices[j - 1];
      matrix->values[j] = matrix->values[j - 1];
      j--;
    }

    // Insert key at correct position
    matrix->row_indices[j] = row_key;
    matrix->col_indices[j] = col_key;
    matrix->values[j] = val_key;
  }
}

int matgen_coo_sort(matgen_coo_matrix_t* matrix) {
  if (!matrix) {
    MATGEN_LOG_ERROR("NULL matrix pointer");
    return -1;
  }

  if (matrix->is_sorted || matrix->nnz <= 1) {
    MATGEN_LOG_DEBUG("Matrix already sorted or trivial (nnz: %zu)",
                     matrix->nnz);
    return 0;
  }

  MATGEN_LOG_DEBUG("Sorting COO matrix with %zu entries", matrix->nnz);

  // TODO: Use quicksort or merge sort for large matrices (nnz > 1000)
  insertion_sort_coo(matrix);

  matrix->is_sorted = true;

  MATGEN_LOG_DEBUG("Matrix sorted successfully");

  return 0;
}

// =============================================================================
// Matrix Information and Access
// =============================================================================

void matgen_coo_print_info(const matgen_coo_matrix_t* matrix, FILE* stream) {
  if (!matrix || !stream) {
    return;
  }

  double sparsity = (matrix->rows * matrix->cols > 0)
                        ? (100.0 * (double)matrix->nnz) /
                              (double)(matrix->rows * matrix->cols)
                        : 0.0;

  fprintf(stream, "COO Matrix Information:\n");
  fprintf(stream, "  Dimensions: %zu x %zu\n", matrix->rows, matrix->cols);
  fprintf(stream, "  Non-zeros:  %zu\n", matrix->nnz);
  fprintf(stream, "  Capacity:   %zu\n", matrix->capacity);
  fprintf(stream, "  Sparsity:   %.4f%%\n", sparsity);
  fprintf(stream, "  Sorted:     %s\n", matrix->is_sorted ? "yes" : "no");
}

double matgen_coo_get(const matgen_coo_matrix_t* matrix, size_t row,
                      size_t col) {
  if (!matrix) {
    MATGEN_LOG_ERROR("NULL matrix pointer");
    return 0.0;
  }

  if (row >= matrix->rows || col >= matrix->cols) {
    MATGEN_LOG_ERROR("Index out of bounds: (%zu, %zu) for %zu x %zu matrix",
                     row, col, matrix->rows, matrix->cols);
    return 0.0;
  }

  // Linear search through entries
  // TODO: If sorted, use binary search
  for (size_t i = 0; i < matrix->nnz; i++) {
    if (matrix->row_indices[i] == row && matrix->col_indices[i] == col) {
      return matrix->values[i];
    }
  }

  return 0.0;  // Not found, assume zero
}

// =============================================================================
// Utility Functions
// =============================================================================

int matgen_coo_reserve(matgen_coo_matrix_t* matrix, size_t capacity) {
  if (!matrix) {
    MATGEN_LOG_ERROR("NULL matrix pointer");
    return -1;
  }

  if (capacity <= matrix->capacity) {
    // Already have enough capacity
    return 0;
  }

  return coo_resize(matrix, capacity);
}

void matgen_coo_clear(matgen_coo_matrix_t* matrix) {
  if (!matrix) {
    return;
  }

  MATGEN_LOG_DEBUG("Clearing COO matrix (was: %zu entries)", matrix->nnz);

  matrix->nnz = 0;
  matrix->is_sorted = true;  // Empty matrix is sorted
}

size_t matgen_coo_memory_usage(const matgen_coo_matrix_t* matrix) {
  if (!matrix) {
    return 0;
  }

  size_t memory = sizeof(matgen_coo_matrix_t);
  memory += matrix->capacity * sizeof(size_t);  // row_indices
  memory += matrix->capacity * sizeof(size_t);  // col_indices
  memory += matrix->capacity * sizeof(double);  // values

  return memory;
}
