#ifndef MATGEN_CORE_MATRIX_OPS_H
#define MATGEN_CORE_MATRIX_OPS_H

/**
 * @file matrix_ops.h
 * @brief Sparse matrix and vector operations
 *
 * Provides fundamental operations for sparse matrices and vectors:
 * - Sparse matrix-vector multiplication (SpMV)
 * - Dense vector operations
 * - Sparse vector operations
 * - Distance metrics for nearest neighbor search
 */

#include <stddef.h>

#include "matgen/core/coo_matrix.h"
#include "matgen/core/csr_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Sparse Matrix-Vector Multiplication
// =============================================================================

/**
 * @brief Sparse matrix-vector multiplication: y = A * x
 *
 * Computes y = A * x where A is sparse (CSR) and x, y are dense.
 *
 * @param A Sparse matrix in CSR format
 * @param x Input vector [cols]
 * @param y Output vector [rows]
 * @return 0 on success, -1 on error
 */
int matgen_csr_matvec(const matgen_csr_matrix_t* A, const double* x, double* y);

/**
 * @brief Sparse matrix-vector multiplication: y = A^T * x
 *
 * Computes y = A^T * x (transpose multiply)
 *
 * @param A Sparse matrix in CSR format
 * @param x Input vector [rows]
 * @param y Output vector [cols]
 * @return 0 on success, -1 on error
 */
int matgen_csr_matvec_transpose(const matgen_csr_matrix_t* A, const double* x,
                                double* y);

/**
 * @brief Sparse matrix-vector multiplication for COO: y = A * x
 *
 * @param A Sparse matrix in COO format
 * @param x Input vector [cols]
 * @param y Output vector [rows] (must be zero-initialized)
 * @return 0 on success, -1 on error
 */
int matgen_coo_matvec(const matgen_coo_matrix_t* A, const double* x, double* y);

// =============================================================================
// Dense Vector Operations
// =============================================================================

/**
 * @brief Vector dot product: result = x^T * y
 *
 * @param x First vector
 * @param y Second vector
 * @param n Vector length
 * @return Dot product
 */
double matgen_vec_dot(const double* x, const double* y, size_t n);

/**
 * @brief Vector 2-norm (Euclidean norm): ||x||_2
 *
 * @param x Vector
 * @param n Vector length
 * @return 2-norm of x
 */
double matgen_vec_norm2(const double* x, size_t n);

/**
 * @brief Vector 1-norm: ||x||_1
 *
 * @param x Vector
 * @param n Vector length
 * @return 1-norm of x
 */
double matgen_vec_norm1(const double* x, size_t n);

/**
 * @brief Vector scaling: y = alpha * x
 *
 * @param alpha Scalar multiplier
 * @param x Input vector
 * @param y Output vector
 * @param n Vector length
 */
void matgen_vec_scale(double alpha, const double* x, double* y, size_t n);

/**
 * @brief Vector addition: z = x + y
 *
 * @param x First vector
 * @param y Second vector
 * @param z Output vector
 * @param n Vector length
 */
void matgen_vec_add(const double* x, const double* y, double* z, size_t n);

/**
 * @brief AXPY operation: y = alpha * x + y
 *
 * @param alpha Scalar multiplier
 * @param x Input vector
 * @param y Input/output vector
 * @param n Vector length
 */
void matgen_vec_axpy(double alpha, const double* x, double* y, size_t n);

// =============================================================================
// Sparse Vector Operations
// =============================================================================

/**
 * @brief Sparse vector dot product
 *
 * Computes dot product of two sparse vectors in coordinate format.
 * Assumes indices are sorted for efficiency.
 *
 * @param idx1 Indices of first sparse vector
 * @param val1 Values of first sparse vector
 * @param nnz1 Number of non-zeros in first vector
 * @param idx2 Indices of second sparse vector
 * @param val2 Values of second sparse vector
 * @param nnz2 Number of non-zeros in second vector
 * @return Dot product
 */
double matgen_sparse_vec_dot(const size_t* idx1, const double* val1,
                             size_t nnz1, const size_t* idx2,
                             const double* val2, size_t nnz2);

/**
 * @brief Sparse vector 2-norm
 *
 * @param val Values array
 * @param nnz Number of non-zeros
 * @return 2-norm
 */
double matgen_sparse_vec_norm2(const double* val, size_t nnz);

// =============================================================================
// Distance Metrics (for Nearest Neighbor Search)
// =============================================================================

/**
 * @brief Euclidean distance between sparse vectors
 *
 * Computes ||x - y||_2 for sparse vectors.
 *
 * @param idx1 Indices of first sparse vector
 * @param val1 Values of first sparse vector
 * @param nnz1 Number of non-zeros in first vector
 * @param idx2 Indices of second sparse vector
 * @param val2 Values of second sparse vector
 * @param nnz2 Number of non-zeros in second vector
 * @return Euclidean distance
 */
double matgen_sparse_euclidean_distance(const size_t* idx1, const double* val1,
                                        size_t nnz1, const size_t* idx2,
                                        const double* val2, size_t nnz2);

/**
 * @brief Cosine distance between sparse vectors
 *
 * Computes 1 - cos(theta) where cos(theta) = (x·y) / (||x|| ||y||)
 * Returns value in [0, 2], where 0 = identical direction, 2 = opposite.
 *
 * @param idx1 Indices of first sparse vector
 * @param val1 Values of first sparse vector
 * @param nnz1 Number of non-zeros in first vector
 * @param idx2 Indices of second sparse vector
 * @param val2 Values of second sparse vector
 * @param nnz2 Number of non-zeros in second vector
 * @return Cosine distance
 */
double matgen_sparse_cosine_distance(const size_t* idx1, const double* val1,
                                     size_t nnz1, const size_t* idx2,
                                     const double* val2, size_t nnz2);

/**
 * @brief Jaccard distance between sparse vectors
 *
 * Computes 1 - |intersection| / |union| based on indices.
 * Treats sparse vectors as sets (ignores values).
 *
 * @param idx1 Indices of first sparse vector
 * @param nnz1 Number of non-zeros in first vector
 * @param idx2 Indices of second sparse vector
 * @param nnz2 Number of non-zeros in second vector
 * @return Jaccard distance in [0, 1]
 */
double matgen_sparse_jaccard_distance(const size_t* idx1, size_t nnz1,
                                      const size_t* idx2, size_t nnz2);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_CORE_MATRIX_OPS_H
