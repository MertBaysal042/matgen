#ifndef MATGEN_MATH_SPARSE_VECTOR_H
#define MATGEN_MATH_SPARSE_VECTOR_H

/**
 * @file sparse_vector.h
 * @brief Sparse vector operations
 *
 * Provides operations for sparse vectors stored in coordinate format
 * (parallel arrays of indices and values):
 * - Dot products
 * - Norms (1-norm and 2-norm)
 *
 * All operations assume indices are sorted for optimal performance.
 * These are useful for:
 * - Row/column operations on sparse matrices
 * - Feature vectors in machine learning
 * - Computing distances between sparse data points
 */

#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Sparse vector dot product
 *
 * Computes dot product of two sparse vectors in coordinate format.
 * Uses merge-like traversal for efficiency.
 *
 * @param idx1 Indices of first sparse vector (sorted)
 * @param val1 Values of first sparse vector
 * @param nnz1 Number of non-zeros in first vector
 * @param idx2 Indices of second sparse vector (sorted)
 * @param val2 Values of second sparse vector
 * @param nnz2 Number of non-zeros in second vector
 * @return Dot product
 *
 * @note Complexity: O(nnz1 + nnz2) with sorted indices
 * @note Assumes indices are sorted in ascending order
 */
matgen_value_t matgen_sparse_vec_dot(
    const matgen_index_t* idx1, const matgen_value_t* val1, matgen_size_t nnz1,
    const matgen_index_t* idx2, const matgen_value_t* val2, matgen_size_t nnz2);

/**
 * @brief Sparse vector 2-norm (Euclidean norm)
 *
 * Computes sqrt(sum(val[i]^2)) for sparse vector.
 *
 * @param val Values array
 * @param nnz Number of non-zeros
 * @return 2-norm
 *
 * @note Complexity: O(nnz)
 */
matgen_value_t matgen_sparse_vec_norm2(const matgen_value_t* val,
                                       matgen_size_t nnz);

/**
 * @brief Sparse vector 1-norm (Manhattan norm)
 *
 * Computes sum(|val[i]|) for sparse vector.
 *
 * @param val Values array
 * @param nnz Number of non-zeros
 * @return 1-norm
 *
 * @note Complexity: O(nnz)
 */
matgen_value_t matgen_sparse_vec_norm1(const matgen_value_t* val,
                                       matgen_size_t nnz);

/**
 * @brief Sparse vector squared 2-norm (without sqrt)
 *
 * Computes sum(val[i]^2) for sparse vector.
 * Faster than norm2 when you don't need the square root.
 *
 * @param val Values array
 * @param nnz Number of non-zeros
 * @return Squared 2-norm
 *
 * @note Complexity: O(nnz)
 * @note Useful for avoiding sqrt when comparing distances
 */
matgen_value_t matgen_sparse_vec_norm2_squared(const matgen_value_t* val,
                                               matgen_size_t nnz);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_MATH_SPARSE_VECTOR_H
