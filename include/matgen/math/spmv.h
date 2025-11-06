#ifndef MATGEN_MATH_SPMV_H
#define MATGEN_MATH_SPMV_H

/**
 * @file spmv.h
 * @brief Sparse matrix-vector multiplication operations
 *
 * Provides SpMV (Sparse Matrix-Vector product) operations for different
 * sparse matrix formats:
 * - CSR format: y = A * x and y = A^T * x
 * - COO format: y = A * x
 *
 * These operations are fundamental building blocks for iterative solvers,
 * eigenvalue computations, and other sparse linear algebra algorithms.
 */

#include "matgen/core/coo_matrix.h"
#include "matgen/core/csr_matrix.h"
#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Sparse matrix-vector multiplication: y = A * x
 *
 * Computes y = A * x where A is sparse (CSR) and x, y are dense.
 * This is the most common SpMV operation.
 *
 * @param A Sparse matrix in CSR format
 * @param x Input vector [cols]
 * @param y Output vector [rows]
 * @return MATGEN_SUCCESS on success, error code otherwise
 *
 * @note Complexity: O(nnz) where nnz is the number of non-zeros in A
 */
matgen_error_t matgen_csr_spmv(const matgen_csr_matrix_t* A,
                               const matgen_value_t* x, matgen_value_t* y);

/**
 * @brief Sparse matrix-vector multiplication: y = A^T * x
 *
 * Computes y = A^T * x (transpose multiply) where A is CSR format.
 * This operation is less cache-friendly than regular SpMV.
 *
 * @param A Sparse matrix in CSR format
 * @param x Input vector [rows]
 * @param y Output vector [cols] (will be zero-initialized internally)
 * @return MATGEN_SUCCESS on success, error code otherwise
 *
 * @note Complexity: O(nnz)
 * @note Output vector y is automatically zeroed before accumulation
 */
matgen_error_t matgen_csr_spmv_transpose(const matgen_csr_matrix_t* A,
                                         const matgen_value_t* x,
                                         matgen_value_t* y);

/**
 * @brief Sparse matrix-vector multiplication for COO: y = A * x
 *
 * Computes y = A * x where A is in COO (coordinate) format.
 * Less efficient than CSR format but useful when matrix is being built.
 *
 * @param A Sparse matrix in COO format
 * @param x Input vector [cols]
 * @param y Output vector [rows] (must be zero-initialized by caller)
 * @return MATGEN_SUCCESS on success, error code otherwise
 *
 * @note Complexity: O(nnz)
 * @note Caller must zero-initialize y before calling this function
 * @note If A has duplicate entries, they will be accumulated in y
 */
matgen_error_t matgen_coo_spmv(const matgen_coo_matrix_t* A,
                               const matgen_value_t* x, matgen_value_t* y);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_MATH_SPMV_H
