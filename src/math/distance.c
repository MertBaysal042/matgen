#include "matgen/math/distance.h"

#include <math.h>

#include "matgen/math/sparse_vector.h"

// =============================================================================
// Euclidean Distance
// =============================================================================

matgen_value_t matgen_sparse_euclidean_distance_squared(
    const matgen_index_t* idx1, const matgen_value_t* val1, matgen_size_t nnz1,
    const matgen_index_t* idx2, const matgen_value_t* val2,
    matgen_size_t nnz2) {
  matgen_value_t sum = 0.0;
  matgen_size_t i = 0;
  matgen_size_t j = 0;

  // Traverse both sparse vectors simultaneously
  while (i < nnz1 && j < nnz2) {
    if (idx1[i] == idx2[j]) {
      // Both have this index
      matgen_value_t diff = val1[i] - val2[j];
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

  return sum;
}

matgen_value_t matgen_sparse_euclidean_distance(const matgen_index_t* idx1,
                                                const matgen_value_t* val1,
                                                matgen_size_t nnz1,
                                                const matgen_index_t* idx2,
                                                const matgen_value_t* val2,
                                                matgen_size_t nnz2) {
  matgen_value_t squared = matgen_sparse_euclidean_distance_squared(
      idx1, val1, nnz1, idx2, val2, nnz2);
  return sqrt(squared);
}

// =============================================================================
// Manhattan Distance
// =============================================================================

matgen_value_t matgen_sparse_manhattan_distance(const matgen_index_t* idx1,
                                                const matgen_value_t* val1,
                                                matgen_size_t nnz1,
                                                const matgen_index_t* idx2,
                                                const matgen_value_t* val2,
                                                matgen_size_t nnz2) {
  matgen_value_t sum = 0.0;
  matgen_size_t i = 0;
  matgen_size_t j = 0;

  // Traverse both sparse vectors simultaneously
  while (i < nnz1 && j < nnz2) {
    if (idx1[i] == idx2[j]) {
      // Both have this index
      sum += fabs(val1[i] - val2[j]);
      i++;
      j++;
    } else if (idx1[i] < idx2[j]) {
      // Only first has this index
      sum += fabs(val1[i]);
      i++;
    } else {
      // Only second has this index
      sum += fabs(val2[j]);
      j++;
    }
  }

  // Remaining entries in first vector
  while (i < nnz1) {
    sum += fabs(val1[i]);
    i++;
  }

  // Remaining entries in second vector
  while (j < nnz2) {
    sum += fabs(val2[j]);
    j++;
  }

  return sum;
}

// =============================================================================
// Cosine Distance
// =============================================================================

matgen_value_t matgen_sparse_cosine_distance(const matgen_index_t* idx1,
                                             const matgen_value_t* val1,
                                             matgen_size_t nnz1,
                                             const matgen_index_t* idx2,
                                             const matgen_value_t* val2,
                                             matgen_size_t nnz2) {
  // Compute dot product and norms
  matgen_value_t dot =
      matgen_sparse_vec_dot(idx1, val1, nnz1, idx2, val2, nnz2);
  matgen_value_t norm1 = matgen_sparse_vec_norm2(val1, nnz1);
  matgen_value_t norm2 = matgen_sparse_vec_norm2(val2, nnz2);

  // Handle zero vectors
  if (norm1 == 0.0 || norm2 == 0.0) {
    return 1.0;  // Maximum cosine distance (orthogonal)
  }

  // Cosine similarity
  matgen_value_t cosine_sim = dot / (norm1 * norm2);

  // Clamp to [-1, 1] to handle numerical errors
  if (cosine_sim > 1.0) {
    cosine_sim = 1.0;
  } else if (cosine_sim < -1.0) {
    cosine_sim = -1.0;
  }

  // Return cosine distance: 1 - similarity
  return 1.0 - cosine_sim;
}

// =============================================================================
// Jaccard Distance
// =============================================================================

matgen_value_t matgen_sparse_jaccard_distance(const matgen_index_t* idx1,
                                              matgen_size_t nnz1,
                                              const matgen_index_t* idx2,
                                              matgen_size_t nnz2) {
  if (nnz1 == 0 && nnz2 == 0) {
    return 0.0;  // Both vectors are empty, define distance as 0
  }

  matgen_size_t intersection = 0;
  matgen_size_t i = 0;
  matgen_size_t j = 0;

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
  matgen_size_t union_size = nnz1 + nnz2 - intersection;

  if (union_size == 0) {
    return 0.0;
  }

  // Jaccard similarity = intersection / union
  matgen_value_t jaccard_sim =
      (matgen_value_t)intersection / (matgen_value_t)union_size;

  // Return Jaccard distance = 1 - similarity
  return 1.0 - jaccard_sim;
}
