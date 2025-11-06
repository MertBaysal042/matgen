#ifndef MATGEN_MATH_DISTANCE_H
#define MATGEN_MATH_DISTANCE_H

/**
 * @file distance.h
 * @brief Distance metrics for sparse vectors
 *
 * Provides various distance and similarity metrics for sparse vectors
 * in coordinate format. These are essential for:
 * - Nearest neighbor search
 * - Clustering algorithms
 * - Similarity-based recommendations
 * - Graph construction from sparse data
 *
 * Supported metrics:
 * - Euclidean (L2) distance
 * - Manhattan (L1) distance
 * - Cosine distance (angular)
 * - Jaccard distance (set-based)
 *
 * All operations assume sorted indices for optimal performance.
 */

#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Euclidean distance between sparse vectors
 *
 * Computes ||x - y||_2 = sqrt(sum((x[i] - y[i])^2)) for sparse vectors.
 * Efficiently handles elements present in only one vector.
 *
 * @param idx1 Indices of first sparse vector (sorted)
 * @param val1 Values of first sparse vector
 * @param nnz1 Number of non-zeros in first vector
 * @param idx2 Indices of second sparse vector (sorted)
 * @param val2 Values of second sparse vector
 * @param nnz2 Number of non-zeros in second vector
 * @return Euclidean distance
 *
 * @note Complexity: O(nnz1 + nnz2)
 * @note Most commonly used distance metric
 */
matgen_value_t matgen_sparse_euclidean_distance(
    const matgen_index_t* idx1, const matgen_value_t* val1, matgen_size_t nnz1,
    const matgen_index_t* idx2, const matgen_value_t* val2, matgen_size_t nnz2);

/**
 * @brief Squared Euclidean distance between sparse vectors
 *
 * Computes ||x - y||_2^2 = sum((x[i] - y[i])^2) without the square root.
 * Faster than euclidean_distance and sufficient for comparisons.
 *
 * @param idx1 Indices of first sparse vector (sorted)
 * @param val1 Values of first sparse vector
 * @param nnz1 Number of non-zeros in first vector
 * @param idx2 Indices of second sparse vector (sorted)
 * @param val2 Values of second sparse vector
 * @param nnz2 Number of non-zeros in second vector
 * @return Squared Euclidean distance
 *
 * @note Complexity: O(nnz1 + nnz2)
 * @note Prefer this over euclidean when comparing distances
 */
matgen_value_t matgen_sparse_euclidean_distance_squared(
    const matgen_index_t* idx1, const matgen_value_t* val1, matgen_size_t nnz1,
    const matgen_index_t* idx2, const matgen_value_t* val2, matgen_size_t nnz2);

/**
 * @brief Manhattan (L1) distance between sparse vectors
 *
 * Computes ||x - y||_1 = sum(|x[i] - y[i]|) for sparse vectors.
 *
 * @param idx1 Indices of first sparse vector (sorted)
 * @param val1 Values of first sparse vector
 * @param nnz1 Number of non-zeros in first vector
 * @param idx2 Indices of second sparse vector (sorted)
 * @param val2 Values of second sparse vector
 * @param nnz2 Number of non-zeros in second vector
 * @return Manhattan distance
 *
 * @note Complexity: O(nnz1 + nnz2)
 * @note More robust to outliers than Euclidean
 */
matgen_value_t matgen_sparse_manhattan_distance(
    const matgen_index_t* idx1, const matgen_value_t* val1, matgen_size_t nnz1,
    const matgen_index_t* idx2, const matgen_value_t* val2, matgen_size_t nnz2);

/**
 * @brief Cosine distance between sparse vectors
 *
 * Computes 1 - cos(theta) where cos(theta) = (x·y) / (||x|| ||y||).
 * Returns value in [0, 2], where:
 * - 0 = identical direction
 * - 1 = orthogonal
 * - 2 = opposite direction
 *
 * @param idx1 Indices of first sparse vector (sorted)
 * @param val1 Values of first sparse vector
 * @param nnz1 Number of non-zeros in first vector
 * @param idx2 Indices of second sparse vector (sorted)
 * @param val2 Values of second sparse vector
 * @param nnz2 Number of non-zeros in second vector
 * @return Cosine distance
 *
 * @note Complexity: O(nnz1 + nnz2)
 * @note Returns 1.0 if either vector is zero
 * @note Invariant to vector magnitude (good for text/TF-IDF)
 */
matgen_value_t matgen_sparse_cosine_distance(
    const matgen_index_t* idx1, const matgen_value_t* val1, matgen_size_t nnz1,
    const matgen_index_t* idx2, const matgen_value_t* val2, matgen_size_t nnz2);

/**
 * @brief Jaccard distance between sparse vectors
 *
 * Computes 1 - |intersection| / |union| based on indices only.
 * Treats sparse vectors as sets (values are ignored).
 * Returns value in [0, 1], where:
 * - 0 = identical sets
 * - 1 = completely disjoint sets
 *
 * @param idx1 Indices of first sparse vector (sorted)
 * @param nnz1 Number of non-zeros in first vector
 * @param idx2 Indices of second sparse vector (sorted)
 * @param nnz2 Number of non-zeros in second vector
 * @return Jaccard distance in [0, 1]
 *
 * @note Complexity: O(nnz1 + nnz2)
 * @note Returns 0.0 if both vectors are empty
 * @note Good for binary/categorical features
 */
matgen_value_t matgen_sparse_jaccard_distance(const matgen_index_t* idx1,
                                              matgen_size_t nnz1,
                                              const matgen_index_t* idx2,
                                              matgen_size_t nnz2);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_MATH_DISTANCE_H
