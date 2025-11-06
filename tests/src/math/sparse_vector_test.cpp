#include <gtest/gtest.h>
#include <matgen/math/sparse_vector.h>

#include <cmath>

const double EPSILON = 1e-10;

// =============================================================================
// Sparse Dot Product Tests
// =============================================================================

TEST(SparseVectorTest, DotProductBasic) {
  // v1 = [0, 2, 0, 4, 0, 6] (indices: 1, 3, 5)
  matgen_index_t idx1[] = {1, 3, 5};
  matgen_value_t val1[] = {2.0, 4.0, 6.0};

  // v2 = [1, 0, 3, 0, 5, 0] (indices: 0, 2, 4)
  matgen_index_t idx2[] = {0, 2, 4};
  matgen_value_t val2[] = {1.0, 3.0, 5.0};

  matgen_value_t result = matgen_sparse_vec_dot(idx1, val1, 3, idx2, val2, 3);

  // No overlapping indices, so dot product is 0
  EXPECT_DOUBLE_EQ(result, 0.0);
}

TEST(SparseVectorTest, DotProductOverlapping) {
  // v1 = [1, 0, 3, 0, 5] (indices: 0, 2, 4)
  matgen_index_t idx1[] = {0, 2, 4};
  matgen_value_t val1[] = {1.0, 3.0, 5.0};

  // v2 = [2, 0, 4, 6, 8] (indices: 0, 2, 3, 4)
  matgen_index_t idx2[] = {0, 2, 3, 4};
  matgen_value_t val2[] = {2.0, 4.0, 6.0, 8.0};

  matgen_value_t result = matgen_sparse_vec_dot(idx1, val1, 3, idx2, val2, 4);

  // Overlapping at indices 0, 2, 4: 1*2 + 3*4 + 5*8 = 2 + 12 + 40 = 54
  EXPECT_DOUBLE_EQ(result, 54.0);
}

TEST(SparseVectorTest, DotProductIdentical) {
  matgen_index_t idx[] = {0, 2, 4};
  matgen_value_t val[] = {1.0, 2.0, 3.0};

  matgen_value_t result = matgen_sparse_vec_dot(idx, val, 3, idx, val, 3);

  // Self dot product: 1^2 + 2^2 + 3^2 = 1 + 4 + 9 = 14
  EXPECT_DOUBLE_EQ(result, 14.0);
}

TEST(SparseVectorTest, DotProductEmpty) {
  matgen_index_t idx[] = {0, 1, 2};
  matgen_value_t val[] = {1.0, 2.0, 3.0};

  matgen_value_t result1 =
      matgen_sparse_vec_dot(idx, val, 3, nullptr, nullptr, 0);
  EXPECT_DOUBLE_EQ(result1, 0.0);

  matgen_value_t result2 =
      matgen_sparse_vec_dot(nullptr, nullptr, 0, idx, val, 3);
  EXPECT_DOUBLE_EQ(result2, 0.0);

  matgen_value_t result3 =
      matgen_sparse_vec_dot(nullptr, nullptr, 0, nullptr, nullptr, 0);
  EXPECT_DOUBLE_EQ(result3, 0.0);
}

TEST(SparseVectorTest, DotProductSingleElement) {
  matgen_index_t idx1[] = {5};
  matgen_value_t val1[] = {3.0};

  matgen_index_t idx2[] = {5};
  matgen_value_t val2[] = {4.0};

  matgen_value_t result = matgen_sparse_vec_dot(idx1, val1, 1, idx2, val2, 1);
  EXPECT_DOUBLE_EQ(result, 12.0);
}

TEST(SparseVectorTest, DotProductSingleElementNoOverlap) {
  matgen_index_t idx1[] = {3};
  matgen_value_t val1[] = {3.0};

  matgen_index_t idx2[] = {5};
  matgen_value_t val2[] = {4.0};

  matgen_value_t result = matgen_sparse_vec_dot(idx1, val1, 1, idx2, val2, 1);
  EXPECT_DOUBLE_EQ(result, 0.0);
}

// =============================================================================
// Sparse Norm Tests
// =============================================================================

TEST(SparseVectorTest, Norm2) {
  matgen_value_t val[] = {3.0, 4.0};

  matgen_value_t norm = matgen_sparse_vec_norm2(val, 2);
  EXPECT_DOUBLE_EQ(norm, 5.0);  // sqrt(9 + 16) = 5
}

TEST(SparseVectorTest, Norm2Single) {
  matgen_value_t val[] = {7.0};

  matgen_value_t norm = matgen_sparse_vec_norm2(val, 1);
  EXPECT_DOUBLE_EQ(norm, 7.0);
}

TEST(SparseVectorTest, Norm2Empty) {
  matgen_value_t norm = matgen_sparse_vec_norm2(nullptr, 0);
  EXPECT_DOUBLE_EQ(norm, 0.0);
}

TEST(SparseVectorTest, Norm2MultipleValues) {
  matgen_value_t val[] = {1.0, 2.0, 3.0, 4.0};

  matgen_value_t norm = matgen_sparse_vec_norm2(val, 4);
  // sqrt(1 + 4 + 9 + 16) = sqrt(30)
  EXPECT_DOUBLE_EQ(norm, sqrt(30.0));
}

TEST(SparseVectorTest, Norm1) {
  matgen_value_t val[] = {1.0, -2.0, 3.0, -4.0};

  matgen_value_t norm = matgen_sparse_vec_norm1(val, 4);
  EXPECT_DOUBLE_EQ(norm, 10.0);  // 1 + 2 + 3 + 4 = 10
}

TEST(SparseVectorTest, Norm1Empty) {
  matgen_value_t norm = matgen_sparse_vec_norm1(nullptr, 0);
  EXPECT_DOUBLE_EQ(norm, 0.0);
}

TEST(SparseVectorTest, Norm2Squared) {
  matgen_value_t val[] = {3.0, 4.0};

  matgen_value_t norm_sq = matgen_sparse_vec_norm2_squared(val, 2);
  EXPECT_DOUBLE_EQ(norm_sq, 25.0);  // 9 + 16 = 25
}

TEST(SparseVectorTest, Norm2SquaredEmpty) {
  matgen_value_t norm_sq = matgen_sparse_vec_norm2_squared(nullptr, 0);
  EXPECT_DOUBLE_EQ(norm_sq, 0.0);
}

// =============================================================================
// Edge Cases and Consistency Tests
// =============================================================================

TEST(SparseVectorTest, Norm2vsNorm2Squared) {
  matgen_value_t val[] = {3.0, 4.0, 5.0, 6.0};

  matgen_value_t norm = matgen_sparse_vec_norm2(val, 4);
  matgen_value_t norm_sq = matgen_sparse_vec_norm2_squared(val, 4);

  EXPECT_DOUBLE_EQ(norm * norm, norm_sq);
}

TEST(SparseVectorTest, DotProductConsistentWithNorm) {
  matgen_index_t idx[] = {0, 2, 5, 7};
  matgen_value_t val[] = {1.0, 2.0, 3.0, 4.0};

  matgen_value_t dot = matgen_sparse_vec_dot(idx, val, 4, idx, val, 4);
  matgen_value_t norm_sq = matgen_sparse_vec_norm2_squared(val, 4);

  EXPECT_DOUBLE_EQ(dot, norm_sq);
}

TEST(SparseVectorTest, LargeSparseVectors) {
  const size_t nnz = 1000;
  matgen_index_t* idx1 = new matgen_index_t[nnz];
  matgen_value_t* val1 = new matgen_value_t[nnz];
  matgen_index_t* idx2 = new matgen_index_t[nnz];
  matgen_value_t* val2 = new matgen_value_t[nnz];

  // Create two sparse vectors with some overlap
  for (size_t i = 0; i < nnz; i++) {
    idx1[i] = i * 2;  // Even indices
    val1[i] = 1.0;
    idx2[i] = (i * 2) + 1;  // Odd indices
    val2[i] = 2.0;
  }

  // No overlap, so dot product should be 0
  matgen_value_t dot = matgen_sparse_vec_dot(idx1, val1, nnz, idx2, val2, nnz);
  EXPECT_DOUBLE_EQ(dot, 0.0);

  // Norms should be non-zero
  matgen_value_t norm1 = matgen_sparse_vec_norm2(val1, nnz);
  matgen_value_t norm2 = matgen_sparse_vec_norm2(val2, nnz);

  EXPECT_DOUBLE_EQ(norm1, sqrt(static_cast<double>(nnz)));
  EXPECT_DOUBLE_EQ(norm2, 2.0 * sqrt(static_cast<double>(nnz)));

  delete[] idx1;
  delete[] val1;
  delete[] idx2;
  delete[] val2;
}
