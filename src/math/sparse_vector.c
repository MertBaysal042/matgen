#include "matgen/math/sparse_vector.h"

#include <math.h>

matgen_value_t matgen_sparse_vec_dot(const matgen_index_t* idx1,
                                     const matgen_value_t* val1,
                                     matgen_size_t nnz1,
                                     const matgen_index_t* idx2,
                                     const matgen_value_t* val2,
                                     matgen_size_t nnz2) {
  matgen_value_t result = 0.0;
  matgen_size_t i = 0;
  matgen_size_t j = 0;

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

matgen_value_t matgen_sparse_vec_norm2(const matgen_value_t* val,
                                       matgen_size_t nnz) {
  matgen_value_t sum = 0.0;
  for (matgen_size_t i = 0; i < nnz; i++) {
    sum += val[i] * val[i];
  }
  return sqrt(sum);
}

matgen_value_t matgen_sparse_vec_norm1(const matgen_value_t* val,
                                       matgen_size_t nnz) {
  matgen_value_t sum = 0.0;
  for (matgen_size_t i = 0; i < nnz; i++) {
    sum += fabs(val[i]);
  }
  return sum;
}

matgen_value_t matgen_sparse_vec_norm2_squared(const matgen_value_t* val,
                                               matgen_size_t nnz) {
  matgen_value_t sum = 0.0;
  for (matgen_size_t i = 0; i < nnz; i++) {
    sum += val[i] * val[i];
  }
  return sum;
}
