#include "matgen/math/spmv.h"

#include <string.h>

// =============================================================================
// CSR SpMV
// =============================================================================

matgen_error_t matgen_csr_spmv(const matgen_csr_matrix_t* A,
                               const matgen_value_t* x, matgen_value_t* y) {
  if (!A || !x || !y) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (A->nnz > 0 && (!A->row_ptr || !A->col_indices || !A->values)) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  // Compute y = A * x
  for (matgen_index_t i = 0; i < A->rows; i++) {
    matgen_value_t sum = 0.0;
    matgen_size_t row_start = A->row_ptr[i];
    matgen_size_t row_end = A->row_ptr[i + 1];

    for (matgen_size_t j = row_start; j < row_end; j++) {
      sum += A->values[j] * x[A->col_indices[j]];
    }

    y[i] = sum;
  }

  return MATGEN_SUCCESS;
}

matgen_error_t matgen_csr_spmv_transpose(const matgen_csr_matrix_t* A,
                                         const matgen_value_t* x,
                                         matgen_value_t* y) {
  if (!A || !x || !y) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (A->nnz > 0 && (!A->row_ptr || !A->col_indices || !A->values)) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  // Zero initialize output
  memset(y, 0, A->cols * sizeof(matgen_value_t));

  // Accumulate: y[col] += A[row,col] * x[row]
  for (matgen_index_t i = 0; i < A->rows; i++) {
    matgen_value_t x_val = x[i];
    matgen_size_t row_start = A->row_ptr[i];
    matgen_size_t row_end = A->row_ptr[i + 1];

    for (matgen_size_t j = row_start; j < row_end; j++) {
      y[A->col_indices[j]] += A->values[j] * x_val;
    }
  }

  return MATGEN_SUCCESS;
}

// =============================================================================
// COO SpMV
// =============================================================================

matgen_error_t matgen_coo_spmv(const matgen_coo_matrix_t* A,
                               const matgen_value_t* x, matgen_value_t* y) {
  if (!A || !x || !y) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (A->nnz > 0 && (!A->row_indices || !A->col_indices || !A->values)) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  // y must be zero-initialized by caller
  // Accumulate: y[row] += A[row,col] * x[col]
  for (matgen_size_t i = 0; i < A->nnz; i++) {
    matgen_index_t row = A->row_indices[i];
    matgen_index_t col = A->col_indices[i];
    y[row] += A->values[i] * x[col];
  }

  return MATGEN_SUCCESS;
}
