#include "matgen/core/matrix_ops.h"

#include <gtest/gtest.h>

#include <cmath>

#include "matgen/core/coo_matrix.h"
#include "matgen/core/csr_matrix.h"
#include "matgen/core/matrix_convert.h"

// =============================================================================
// SpMV Tests
// =============================================================================

TEST(MatrixOps, CSRMatvecIdentity) {
  // Create 3x3 identity matrix
  matgen_coo_matrix_t* coo = matgen_coo_create(3, 3, 3);
  matgen_coo_add_entry(coo, 0, 0, 1.0);
  matgen_coo_add_entry(coo, 1, 1, 1.0);
  matgen_coo_add_entry(coo, 2, 2, 1.0);

  matgen_csr_matrix_t* csr = matgen_coo_to_csr(coo);
  ASSERT_NE(csr, nullptr);

  double x[3] = {1.0, 2.0, 3.0};
  double y[3];

  ASSERT_EQ(matgen_csr_matvec(csr, x, y), 0);

  EXPECT_DOUBLE_EQ(y[0], 1.0);
  EXPECT_DOUBLE_EQ(y[1], 2.0);
  EXPECT_DOUBLE_EQ(y[2], 3.0);

  matgen_coo_destroy(coo);
  matgen_csr_destroy(csr);
}

TEST(MatrixOps, CSRMatvecGeneral) {
  // Create matrix:
  // [ 1  2  0 ]
  // [ 0  3  4 ]
  // [ 5  0  6 ]

  matgen_coo_matrix_t* coo = matgen_coo_create(3, 3, 6);
  matgen_coo_add_entry(coo, 0, 0, 1.0);
  matgen_coo_add_entry(coo, 0, 1, 2.0);
  matgen_coo_add_entry(coo, 1, 1, 3.0);
  matgen_coo_add_entry(coo, 1, 2, 4.0);
  matgen_coo_add_entry(coo, 2, 0, 5.0);
  matgen_coo_add_entry(coo, 2, 2, 6.0);

  matgen_csr_matrix_t* csr = matgen_coo_to_csr(coo);

  double x[3] = {1.0, 2.0, 3.0};
  double y[3];

  matgen_csr_matvec(csr, x, y);

  // y[0] = 1*1 + 2*2 + 0*3 = 5
  // y[1] = 0*1 + 3*2 + 4*3 = 18
  // y[2] = 5*1 + 0*2 + 6*3 = 23

  EXPECT_DOUBLE_EQ(y[0], 5.0);
  EXPECT_DOUBLE_EQ(y[1], 18.0);
  EXPECT_DOUBLE_EQ(y[2], 23.0);

  matgen_coo_destroy(coo);
  matgen_csr_destroy(csr);
}

TEST(MatrixOps, CSRMatvecTranspose) {
  // Create matrix:
  // [ 1  2 ]
  // [ 3  4 ]
  // [ 5  6 ]

  matgen_coo_matrix_t* coo = matgen_coo_create(3, 2, 6);
  matgen_coo_add_entry(coo, 0, 0, 1.0);
  matgen_coo_add_entry(coo, 0, 1, 2.0);
  matgen_coo_add_entry(coo, 1, 0, 3.0);
  matgen_coo_add_entry(coo, 1, 1, 4.0);
  matgen_coo_add_entry(coo, 2, 0, 5.0);
  matgen_coo_add_entry(coo, 2, 1, 6.0);

  matgen_csr_matrix_t* csr = matgen_coo_to_csr(coo);

  double x[3] = {1.0, 2.0, 3.0};
  double y[2];

  matgen_csr_matvec_transpose(csr, x, y);

  // A^T * x where A^T is 2x3
  // y[0] = 1*1 + 3*2 + 5*3 = 22
  // y[1] = 2*1 + 4*2 + 6*3 = 28

  EXPECT_DOUBLE_EQ(y[0], 22.0);
  EXPECT_DOUBLE_EQ(y[1], 28.0);

  matgen_coo_destroy(coo);
  matgen_csr_destroy(csr);
}

TEST(MatrixOps, COOMatvec) {
  matgen_coo_matrix_t* coo = matgen_coo_create(2, 2, 3);
  matgen_coo_add_entry(coo, 0, 0, 2.0);
  matgen_coo_add_entry(coo, 0, 1, 3.0);
  matgen_coo_add_entry(coo, 1, 1, 4.0);

  double x[2] = {1.0, 2.0};
  double y[2] = {0.0, 0.0};

  matgen_coo_matvec(coo, x, y);

  EXPECT_DOUBLE_EQ(y[0], 8.0);  // 2*1 + 3*2
  EXPECT_DOUBLE_EQ(y[1], 8.0);  // 4*2

  matgen_coo_destroy(coo);
}

// =============================================================================
// Dense Vector Operations Tests
// =============================================================================

TEST(MatrixOps, VecDot) {
  double x[3] = {1.0, 2.0, 3.0};
  double y[3] = {4.0, 5.0, 6.0};

  double result = matgen_vec_dot(x, y, 3);

  // 1*4 + 2*5 + 3*6 = 32
  EXPECT_DOUBLE_EQ(result, 32.0);
}

TEST(MatrixOps, VecNorm2) {
  double x[3] = {3.0, 4.0, 0.0};

  double norm = matgen_vec_norm2(x, 3);

  EXPECT_DOUBLE_EQ(norm, 5.0);  // sqrt(9 + 16) = 5
}

TEST(MatrixOps, VecNorm1) {
  double x[3] = {1.0, -2.0, 3.0};

  double norm = matgen_vec_norm1(x, 3);

  EXPECT_DOUBLE_EQ(norm, 6.0);  // |1| + |-2| + |3| = 6
}

TEST(MatrixOps, VecScale) {
  double x[3] = {1.0, 2.0, 3.0};
  double y[3];

  matgen_vec_scale(2.0, x, y, 3);

  EXPECT_DOUBLE_EQ(y[0], 2.0);
  EXPECT_DOUBLE_EQ(y[1], 4.0);
  EXPECT_DOUBLE_EQ(y[2], 6.0);
}

TEST(MatrixOps, VecAdd) {
  double x[3] = {1.0, 2.0, 3.0};
  double y[3] = {4.0, 5.0, 6.0};
  double z[3];

  matgen_vec_add(x, y, z, 3);

  EXPECT_DOUBLE_EQ(z[0], 5.0);
  EXPECT_DOUBLE_EQ(z[1], 7.0);
  EXPECT_DOUBLE_EQ(z[2], 9.0);
}

TEST(MatrixOps, VecAXPY) {
  double x[3] = {1.0, 2.0, 3.0};
  double y[3] = {4.0, 5.0, 6.0};

  matgen_vec_axpy(2.0, x, y, 3);

  // y = 2*x + y
  EXPECT_DOUBLE_EQ(y[0], 6.0);   // 2*1 + 4
  EXPECT_DOUBLE_EQ(y[1], 9.0);   // 2*2 + 5
  EXPECT_DOUBLE_EQ(y[2], 12.0);  // 2*3 + 6
}

// =============================================================================
// Sparse Vector Operations Tests
// =============================================================================

TEST(MatrixOps, SparseVecDot) {
  size_t idx1[3] = {0, 2, 4};
  double val1[3] = {1.0, 2.0, 3.0};

  size_t idx2[3] = {1, 2, 4};
  double val2[3] = {4.0, 5.0, 6.0};

  double result = matgen_sparse_vec_dot(idx1, val1, 3, idx2, val2, 3);

  // Common indices: 2 and 4
  // 2*5 + 3*6 = 28
  EXPECT_DOUBLE_EQ(result, 28.0);
}

TEST(MatrixOps, SparseVecDotDisjoint) {
  size_t idx1[2] = {0, 2};
  double val1[2] = {1.0, 2.0};

  size_t idx2[2] = {1, 3};
  double val2[2] = {3.0, 4.0};

  double result = matgen_sparse_vec_dot(idx1, val1, 2, idx2, val2, 2);

  // No common indices
  EXPECT_DOUBLE_EQ(result, 0.0);
}

TEST(MatrixOps, SparseVecNorm2) {
  double val[3] = {3.0, 4.0, 0.0};

  double norm = matgen_sparse_vec_norm2(val, 3);

  EXPECT_DOUBLE_EQ(norm, 5.0);
}

// =============================================================================
// Distance Metrics Tests
// =============================================================================

TEST(MatrixOps, EuclideanDistanceIdentical) {
  size_t idx[3] = {0, 2, 4};
  double val[3] = {1.0, 2.0, 3.0};

  double dist = matgen_sparse_euclidean_distance(idx, val, 3, idx, val, 3);

  EXPECT_NEAR(dist, 0.0, 1e-10);
}

TEST(MatrixOps, EuclideanDistanceGeneral) {
  size_t idx1[3] = {0, 2, 4};
  double val1[3] = {1.0, 2.0, 3.0};

  size_t idx2[3] = {0, 2, 4};
  double val2[3] = {4.0, 5.0, 6.0};

  double dist = matgen_sparse_euclidean_distance(idx1, val1, 3, idx2, val2, 3);

  // sqrt((1-4)^2 + (2-5)^2 + (3-6)^2) = sqrt(9+9+9) = sqrt(27)
  EXPECT_NEAR(dist, sqrt(27.0), 1e-10);
}

TEST(MatrixOps, EuclideanDistanceDifferentSupport) {
  size_t idx1[2] = {0, 2};
  double val1[2] = {1.0, 2.0};

  size_t idx2[2] = {2, 4};
  double val2[2] = {3.0, 4.0};

  double dist = matgen_sparse_euclidean_distance(idx1, val1, 2, idx2, val2, 2);

  // Differences: (1-0)^2 + (2-3)^2 + (0-4)^2 = 1 + 1 + 16 = 18
  EXPECT_NEAR(dist, sqrt(18.0), 1e-10);
}

TEST(MatrixOps, CosineDistanceOrthogonal) {
  size_t idx1[2] = {0, 2};
  double val1[2] = {1.0, 0.0};

  size_t idx2[2] = {1, 3};
  double val2[2] = {0.0, 1.0};

  double dist = matgen_sparse_cosine_distance(idx1, val1, 2, idx2, val2, 2);

  // Orthogonal vectors: cosine similarity = 0, distance = 1
  EXPECT_NEAR(dist, 1.0, 1e-10);
}

TEST(MatrixOps, CosineDistanceParallel) {
  size_t idx[2] = {0, 2};
  double val1[2] = {1.0, 2.0};
  double val2[2] = {2.0, 4.0};  // Parallel to val1

  double dist = matgen_sparse_cosine_distance(idx, val1, 2, idx, val2, 2);

  // Parallel vectors: cosine similarity = 1, distance = 0
  EXPECT_NEAR(dist, 0.0, 1e-10);
}

TEST(MatrixOps, CosineDistanceOpposite) {
  size_t idx[2] = {0, 2};
  double val1[2] = {1.0, 2.0};
  double val2[2] = {-1.0, -2.0};  // Opposite to val1

  double dist = matgen_sparse_cosine_distance(idx, val1, 2, idx, val2, 2);

  // Opposite vectors: cosine similarity = -1, distance = 2
  EXPECT_NEAR(dist, 2.0, 1e-10);
}

TEST(MatrixOps, JaccardDistanceIdentical) {
  size_t idx[3] = {0, 2, 4};

  double dist = matgen_sparse_jaccard_distance(idx, 3, idx, 3);

  // Identical sets: Jaccard = 1, distance = 0
  EXPECT_DOUBLE_EQ(dist, 0.0);
}

TEST(MatrixOps, JaccardDistanceDisjoint) {
  size_t idx1[2] = {0, 2};
  size_t idx2[2] = {1, 3};

  double dist = matgen_sparse_jaccard_distance(idx1, 2, idx2, 2);

  // Disjoint sets: Jaccard = 0, distance = 1
  EXPECT_DOUBLE_EQ(dist, 1.0);
}

TEST(MatrixOps, JaccardDistancePartialOverlap) {
  size_t idx1[3] = {0, 2, 4};
  size_t idx2[3] = {2, 3, 4};

  double dist = matgen_sparse_jaccard_distance(idx1, 3, idx2, 3);

  // Intersection: {2, 4} = 2
  // Union: {0, 2, 3, 4} = 4
  // Jaccard = 2/4 = 0.5, distance = 0.5
  EXPECT_DOUBLE_EQ(dist, 0.5);
}
