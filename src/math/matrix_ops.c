#include "matgen/math/matrix_ops.h"

#include <math.h>
#include <string.h>

// =============================================================================
// Sparse Matrix-Vector Multiplication
// =============================================================================

int matgen_csr_matvec(const matgen_csr_matrix_t* A, const double* x,
                      double* y) {
  if (!A || !x || !y) {
    return -1;
  }

  for (size_t i = 0; i < A->rows; i++) {
    double sum = 0.0;
    size_t row_start = A->row_ptr[i];
    size_t row_end = A->row_ptr[i + 1];

    for (size_t j = row_start; j < row_end; j++) {
      sum += A->values[j] * x[A->col_indices[j]];
    }

    y[i] = sum;
  }

  return 0;
}

int matgen_csr_matvec_transpose(const matgen_csr_matrix_t* A, const double* x,
                                double* y) {
  if (!A || !x || !y) {
    return -1;
  }

  // Zero initialize output
  memset(y, 0, A->cols * sizeof(double));

  // Accumulate: y[col] += A[row,col] * x[row]
  for (size_t i = 0; i < A->rows; i++) {
    double x_val = x[i];
    size_t row_start = A->row_ptr[i];
    size_t row_end = A->row_ptr[i + 1];

    for (size_t j = row_start; j < row_end; j++) {
      y[A->col_indices[j]] += A->values[j] * x_val;
    }
  }

  return 0;
}

int matgen_coo_matvec(const matgen_coo_matrix_t* A, const double* x,
                      double* y) {
  if (!A || !x || !y) {
    return -1;
  }

  // y must be zero-initialized by caller

  for (size_t i = 0; i < A->nnz; i++) {
    size_t row = A->row_indices[i];
    size_t col = A->col_indices[i];
    y[row] += A->values[i] * x[col];
  }

  return 0;
}

// =============================================================================
// Dense Vector Operations
// =============================================================================

double matgen_vec_dot(const double* x, const double* y, size_t n) {
  double result = 0.0;
  for (size_t i = 0; i < n; i++) {
    result += x[i] * y[i];
  }
  return result;
}

double matgen_vec_norm2(const double* x, size_t n) {
  double sum = 0.0;
  for (size_t i = 0; i < n; i++) {
    sum += x[i] * x[i];
  }
  return sqrt(sum);
}

double matgen_vec_norm1(const double* x, size_t n) {
  double sum = 0.0;
  for (size_t i = 0; i < n; i++) {
    sum += fabs(x[i]);
  }
  return sum;
}

void matgen_vec_scale(double alpha, const double* x, double* y, size_t n) {
  for (size_t i = 0; i < n; i++) {
    y[i] = alpha * x[i];
  }
}

void matgen_vec_add(const double* x, const double* y, double* z, size_t n) {
  for (size_t i = 0; i < n; i++) {
    z[i] = x[i] + y[i];
  }
}

void matgen_vec_axpy(double alpha, const double* x, double* y, size_t n) {
  for (size_t i = 0; i < n; i++) {
    y[i] += alpha * x[i];
  }
}

// =============================================================================
// Sparse Vector Operations
// =============================================================================

double matgen_sparse_vec_dot(const size_t* idx1, const double* val1,
                             size_t nnz1, const size_t* idx2,
                             const double* val2, size_t nnz2) {
  double result = 0.0;
  size_t i = 0;
  size_t j = 0;

  // Merge-like traversal (assumes sorted indices)
  while (i < nnz1 && j < nnz2) {
    if (idx1[i] == idx2[j]) {
      result += val1[i] * val2[j];
      i++;
      j++;
    } else if (idx1[i] < idx2[j]) {
      i++;
    } else {
      j++;
    }
  }

  return result;
}

double matgen_sparse_vec_norm2(const double* val, size_t nnz) {
  double sum = 0.0;
  for (size_t i = 0; i < nnz; i++) {
    sum += val[i] * val[i];
  }
  return sqrt(sum);
}

// =============================================================================
// Distance Metrics
// =============================================================================

double matgen_sparse_euclidean_distance(const size_t* idx1, const double* val1,
                                        size_t nnz1, const size_t* idx2,
                                        const double* val2, size_t nnz2) {
  double sum = 0.0;
  size_t i = 0;
  size_t j = 0;

  // Traverse both sparse vectors simultaneously
  while (i < nnz1 && j < nnz2) {
    if (idx1[i] == idx2[j]) {
      // Both have this index
      double diff = val1[i] - val2[j];
      sum += diff * diff;
      i++;
      j++;
    } else if (idx1[i] < idx2[j]) {
      // Only first has this index
      sum += val1[i] * val1[i];
      i++;
    } else {
      // Only second has this index
      sum += val2[j] * val2[j];
      j++;
    }
  }

  // Remaining entries in first vector
  while (i < nnz1) {
    sum += val1[i] * val1[i];
    i++;
  }

  // Remaining entries in second vector
  while (j < nnz2) {
    sum += val2[j] * val2[j];
    j++;
  }

  return sqrt(sum);
}

double matgen_sparse_cosine_distance(const size_t* idx1, const double* val1,
                                     size_t nnz1, const size_t* idx2,
                                     const double* val2, size_t nnz2) {
  // Compute dot product and norms
  double dot = matgen_sparse_vec_dot(idx1, val1, nnz1, idx2, val2, nnz2);
  double norm1 = matgen_sparse_vec_norm2(val1, nnz1);
  double norm2 = matgen_sparse_vec_norm2(val2, nnz2);

  // Handle zero vectors
  if (norm1 == 0.0 || norm2 == 0.0) {
    return 1.0;  // Maximum cosine distance
  }

  // Cosine similarity
  double cosine_sim = dot / (norm1 * norm2);

  // Clamp to [-1, 1] to handle numerical errors
  if (cosine_sim > 1.0) {
    cosine_sim = 1.0;
  } else if (cosine_sim < -1.0) {
    cosine_sim = -1.0;
  }

  // Return cosine distance: 1 - similarity
  return 1.0 - cosine_sim;
}

double matgen_sparse_jaccard_distance(const size_t* idx1, size_t nnz1,
                                      const size_t* idx2, size_t nnz2) {
  if (nnz1 == 0 && nnz2 == 0) {
    return 0.0;  // Both vectors are empty, define distance as 0
  }

  size_t intersection = 0;
  size_t i = 0;
  size_t j = 0;

  // Count intersection (assumes sorted indices)
  while (i < nnz1 && j < nnz2) {
    if (idx1[i] == idx2[j]) {
      intersection++;
      i++;
      j++;
    } else if (idx1[i] < idx2[j]) {
      i++;
    } else {
      j++;
    }
  }

  // Union size = nnz1 + nnz2 - intersection
  size_t union_size = nnz1 + nnz2 - intersection;

  if (union_size == 0) {
    return 0.0;
  }

  // Jaccard similarity = intersection / union
  double jaccard_sim = (double)intersection / (double)union_size;

  // Return Jaccard distance = 1 - similarity
  return 1.0 - jaccard_sim;
}
