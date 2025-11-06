#ifndef MATGEN_MATH_VECTOR_OPS_H
#define MATGEN_MATH_VECTOR_OPS_H

/**
 * @file vector_ops.h
 * @brief Dense vector operations (BLAS-like)
 *
 * Provides fundamental dense vector operations similar to BLAS Level 1:
 * - Dot products and norms
 * - Vector scaling, addition, subtraction
 * - AXPY operations
 * - Copy and initialization
 *
 * All operations are optimized for cache efficiency and can be parallelized.
 */

#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Norms and Dot Products
// =============================================================================

/**
 * @brief Vector dot product: result = x^T * y
 *
 * Computes the inner product of two vectors.
 *
 * @param x First vector
 * @param y Second vector
 * @param n Vector length
 * @return Dot product sum(x[i] * y[i])
 *
 * @note Complexity: O(n)
 */
matgen_value_t matgen_vec_dot(const matgen_value_t* x, const matgen_value_t* y,
                              matgen_size_t n);

/**
 * @brief Vector 2-norm (Euclidean norm): ||x||_2
 *
 * Computes sqrt(sum(x[i]^2)).
 *
 * @param x Vector
 * @param n Vector length
 * @return 2-norm of x
 *
 * @note Complexity: O(n)
 */
matgen_value_t matgen_vec_norm2(const matgen_value_t* x, matgen_size_t n);

/**
 * @brief Vector 1-norm (Manhattan norm): ||x||_1
 *
 * Computes sum(|x[i]|).
 *
 * @param x Vector
 * @param n Vector length
 * @return 1-norm of x
 *
 * @note Complexity: O(n)
 */
matgen_value_t matgen_vec_norm1(const matgen_value_t* x, matgen_size_t n);

/**
 * @brief Vector infinity norm: ||x||_inf
 *
 * Computes max(|x[i]|).
 *
 * @param x Vector
 * @param n Vector length
 * @return Infinity norm of x
 *
 * @note Complexity: O(n)
 */
matgen_value_t matgen_vec_norminf(const matgen_value_t* x, matgen_size_t n);

// =============================================================================
// Basic Vector Operations
// =============================================================================

/**
 * @brief Vector scaling: y = alpha * x
 *
 * Scales vector x by scalar alpha.
 *
 * @param alpha Scalar multiplier
 * @param x Input vector
 * @param y Output vector (can be same as x for in-place operation)
 * @param n Vector length
 *
 * @note Complexity: O(n)
 * @note Supports in-place operation when x == y
 */
void matgen_vec_scale(matgen_value_t alpha, const matgen_value_t* x,
                      matgen_value_t* y, matgen_size_t n);

/**
 * @brief Vector addition: z = x + y
 *
 * Element-wise addition of two vectors.
 *
 * @param x First vector
 * @param y Second vector
 * @param z Output vector (can alias x or y for in-place)
 * @param n Vector length
 *
 * @note Complexity: O(n)
 * @note Supports in-place operation when z aliases x or y
 */
void matgen_vec_add(const matgen_value_t* x, const matgen_value_t* y,
                    matgen_value_t* z, matgen_size_t n);

/**
 * @brief Vector subtraction: z = x - y
 *
 * Element-wise subtraction of two vectors.
 *
 * @param x First vector (minuend)
 * @param y Second vector (subtrahend)
 * @param z Output vector (can alias x or y for in-place)
 * @param n Vector length
 *
 * @note Complexity: O(n)
 * @note Supports in-place operation when z aliases x or y
 */
void matgen_vec_sub(const matgen_value_t* x, const matgen_value_t* y,
                    matgen_value_t* z, matgen_size_t n);

/**
 * @brief AXPY operation: y = alpha * x + y
 *
 * Scaled vector addition (BLAS saxpy/daxpy equivalent).
 * One of the most important BLAS Level 1 operations.
 *
 * @param alpha Scalar multiplier
 * @param x Input vector
 * @param y Input/output vector (modified in-place)
 * @param n Vector length
 *
 * @note Complexity: O(n)
 * @note y is modified in-place
 */
void matgen_vec_axpy(matgen_value_t alpha, const matgen_value_t* x,
                     matgen_value_t* y, matgen_size_t n);

/**
 * @brief AXPBY operation: z = alpha * x + beta * y
 *
 * Scaled vector addition with two scale factors.
 *
 * @param alpha Scalar multiplier for x
 * @param x First input vector
 * @param beta Scalar multiplier for y
 * @param y Second input vector
 * @param z Output vector (can alias x or y)
 * @param n Vector length
 *
 * @note Complexity: O(n)
 */
void matgen_vec_axpby(matgen_value_t alpha, const matgen_value_t* x,
                      matgen_value_t beta, const matgen_value_t* y,
                      matgen_value_t* z, matgen_size_t n);

// =============================================================================
// Utility Operations
// =============================================================================

/**
 * @brief Copy vector: y = x
 *
 * Copies contents of x into y.
 *
 * @param x Source vector
 * @param y Destination vector
 * @param n Vector length
 *
 * @note Complexity: O(n)
 * @note Uses optimized memcpy
 */
void matgen_vec_copy(const matgen_value_t* x, matgen_value_t* y,
                     matgen_size_t n);

/**
 * @brief Zero-initialize vector: x = 0
 *
 * Sets all elements of vector to zero.
 *
 * @param x Vector to zero
 * @param n Vector length
 *
 * @note Complexity: O(n)
 * @note Uses optimized memset
 */
void matgen_vec_zero(matgen_value_t* x, matgen_size_t n);

/**
 * @brief Fill vector with constant: x = alpha
 *
 * Sets all elements of vector to the same value.
 *
 * @param x Vector to fill
 * @param alpha Value to fill with
 * @param n Vector length
 *
 * @note Complexity: O(n)
 */
void matgen_vec_fill(matgen_value_t* x, matgen_value_t alpha, matgen_size_t n);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_MATH_VECTOR_OPS_H
