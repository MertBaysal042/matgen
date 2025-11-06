#include <gtest/gtest.h>
#include <matgen/math/distance.h>

#include <cmath>

const double EPSILON = 1e-10;

// =============================================================================
// Euclidean Distance Tests
// =============================================================================

TEST(DistanceTest, EuclideanDistanceBasic) {
  // v1 = [1, 0, 2, 0, 3] (indices: 0, 2, 4)
  matgen_index_t idx1[] = {0, 2, 4};
  matgen_value_t val1[] = {1.0, 2.0, 3.0};

  // v2 = [4, 0, 5, 0, 6] (indices: 0, 2, 4)
  matgen_index_t idx2[] = {0, 2, 4};
  matgen_value_t val2[] = {4.0, 5.0, 6.0};

  matgen_value_t dist =
      matgen_sparse_euclidean_distance(idx1, val1, 3, idx2, val2, 3);

  // sqrt((1-4)^2 + (2-5)^2 + (3-6)^2) = sqrt(9 + 9 + 9) = sqrt(27)
  EXPECT_DOUBLE_EQ(dist, sqrt(27.0));
}

TEST(DistanceTest, EuclideanDistanceSquared) {
  matgen_index_t idx1[] = {0, 2, 4};
  matgen_value_t val1[] = {1.0, 2.0, 3.0};

  matgen_index_t idx2[] = {0, 2, 4};
  matgen_value_t val2[] = {4.0, 5.0, 6.0};

  matgen_value_t dist_sq =
      matgen_sparse_euclidean_distance_squared(idx1, val1, 3, idx2, val2, 3);

  EXPECT_DOUBLE_EQ(dist_sq, 27.0);
}

TEST(DistanceTest, EuclideanDistanceIdentical) {
  matgen_index_t idx[] = {1, 3, 5};
  matgen_value_t val[] = {2.0, 4.0, 6.0};

  matgen_value_t dist =
      matgen_sparse_euclidean_distance(idx, val, 3, idx, val, 3);

  EXPECT_DOUBLE_EQ(dist, 0.0);
}

TEST(DistanceTest, EuclideanDistanceNoOverlap) {
  // v1 = [1, 2, 0, 0] (indices: 0, 1)
  matgen_index_t idx1[] = {0, 1};
  matgen_value_t val1[] = {1.0, 2.0};

  // v2 = [0, 0, 3, 4] (indices: 2, 3)
  matgen_index_t idx2[] = {2, 3};
  matgen_value_t val2[] = {3.0, 4.0};

  matgen_value_t dist =
      matgen_sparse_euclidean_distance(idx1, val1, 2, idx2, val2, 2);

  // sqrt(1^2 + 2^2 + 3^2 + 4^2) = sqrt(30)
  EXPECT_DOUBLE_EQ(dist, sqrt(30.0));
}

TEST(DistanceTest, EuclideanDistanceEmpty) {
  matgen_index_t idx[] = {0, 1};
  matgen_value_t val[] = {1.0, 2.0};

  matgen_value_t dist1 =
      matgen_sparse_euclidean_distance(idx, val, 2, nullptr, nullptr, 0);
  EXPECT_DOUBLE_EQ(dist1, sqrt(5.0));  // sqrt(1 + 4)

  matgen_value_t dist2 =
      matgen_sparse_euclidean_distance(nullptr, nullptr, 0, idx, val, 2);
  EXPECT_DOUBLE_EQ(dist2, sqrt(5.0));

  matgen_value_t dist3 = matgen_sparse_euclidean_distance(nullptr, nullptr, 0,
                                                          nullptr, nullptr, 0);
  EXPECT_DOUBLE_EQ(dist3, 0.0);
}

// =============================================================================
// Manhattan Distance Tests
// =============================================================================

TEST(DistanceTest, ManhattanDistanceBasic) {
  matgen_index_t idx1[] = {0, 2, 4};
  matgen_value_t val1[] = {1.0, 2.0, 3.0};

  matgen_index_t idx2[] = {0, 2, 4};
  matgen_value_t val2[] = {4.0, 5.0, 6.0};

  matgen_value_t dist =
      matgen_sparse_manhattan_distance(idx1, val1, 3, idx2, val2, 3);

  // |1-4| + |2-5| + |3-6| = 3 + 3 + 3 = 9
  EXPECT_DOUBLE_EQ(dist, 9.0);
}

TEST(DistanceTest, ManhattanDistanceIdentical) {
  matgen_index_t idx[] = {1, 3, 5};
  matgen_value_t val[] = {2.0, 4.0, 6.0};

  matgen_value_t dist =
      matgen_sparse_manhattan_distance(idx, val, 3, idx, val, 3);

  EXPECT_DOUBLE_EQ(dist, 0.0);
}

TEST(DistanceTest, ManhattanDistanceNoOverlap) {
  matgen_index_t idx1[] = {0, 1};
  matgen_value_t val1[] = {1.0, 2.0};

  matgen_index_t idx2[] = {2, 3};
  matgen_value_t val2[] = {3.0, 4.0};

  matgen_value_t dist =
      matgen_sparse_manhattan_distance(idx1, val1, 2, idx2, val2, 2);

  // 1 + 2 + 3 + 4 = 10
  EXPECT_DOUBLE_EQ(dist, 10.0);
}

TEST(DistanceTest, ManhattanDistanceWithNegatives) {
  matgen_index_t idx1[] = {0, 1, 2};
  matgen_value_t val1[] = {-1.0, 2.0, -3.0};

  matgen_index_t idx2[] = {0, 1, 2};
  matgen_value_t val2[] = {1.0, -2.0, 3.0};

  matgen_value_t dist =
      matgen_sparse_manhattan_distance(idx1, val1, 3, idx2, val2, 3);

  // |-1-1| + |2-(-2)| + |-3-3| = 2 + 4 + 6 = 12
  EXPECT_DOUBLE_EQ(dist, 12.0);
}

// =============================================================================
// Cosine Distance Tests
// =============================================================================

TEST(DistanceTest, CosineDistanceOrthogonal) {
  // v1 = [1, 0] (index: 0)
  matgen_index_t idx1[] = {0};
  matgen_value_t val1[] = {1.0};

  // v2 = [0, 1] (index: 1)
  matgen_index_t idx2[] = {1};
  matgen_value_t val2[] = {1.0};

  matgen_value_t dist =
      matgen_sparse_cosine_distance(idx1, val1, 1, idx2, val2, 1);

  // Orthogonal vectors: cosine similarity = 0, distance = 1
  EXPECT_DOUBLE_EQ(dist, 1.0);
}

TEST(DistanceTest, CosineDistanceIdentical) {
  matgen_index_t idx[] = {0, 1, 2};
  matgen_value_t val[] = {1.0, 2.0, 3.0};

  matgen_value_t dist = matgen_sparse_cosine_distance(idx, val, 3, idx, val, 3);

  // Identical vectors: cosine similarity = 1, distance = 0
  EXPECT_NEAR(dist, 0.0, EPSILON);
}

TEST(DistanceTest, CosineDistanceOpposite) {
  matgen_index_t idx[] = {0, 1};
  matgen_value_t val1[] = {1.0, 2.0};
  matgen_value_t val2[] = {-1.0, -2.0};

  matgen_value_t dist =
      matgen_sparse_cosine_distance(idx, val1, 2, idx, val2, 2);

  // Opposite vectors: cosine similarity = -1, distance = 2
  EXPECT_NEAR(dist, 2.0, EPSILON);
}

TEST(DistanceTest, CosineDistanceParallel) {
  matgen_index_t idx[] = {0, 1, 2};
  matgen_value_t val1[] = {1.0, 2.0, 3.0};
  matgen_value_t val2[] = {2.0, 4.0, 6.0};

  matgen_value_t dist =
      matgen_sparse_cosine_distance(idx, val1, 3, idx, val2, 3);

  // Parallel vectors (same direction): distance = 0
  EXPECT_NEAR(dist, 0.0, EPSILON);
}

TEST(DistanceTest, CosineDistanceZeroVector) {
  matgen_index_t idx1[] = {0};
  matgen_value_t val1[] = {1.0};

  matgen_value_t dist =
      matgen_sparse_cosine_distance(idx1, val1, 1, nullptr, nullptr, 0);

  // Zero vector: returns 1.0 (maximum distance)
  EXPECT_DOUBLE_EQ(dist, 1.0);
}

// =============================================================================
// Jaccard Distance Tests
// =============================================================================

TEST(DistanceTest, JaccardDistanceIdentical) {
  matgen_index_t idx[] = {0, 2, 4, 6};

  matgen_value_t dist = matgen_sparse_jaccard_distance(idx, 4, idx, 4);

  // Identical sets: distance = 0
  EXPECT_DOUBLE_EQ(dist, 0.0);
}

TEST(DistanceTest, JaccardDistanceDisjoint) {
  matgen_index_t idx1[] = {0, 1, 2};
  matgen_index_t idx2[] = {3, 4, 5};

  matgen_value_t dist = matgen_sparse_jaccard_distance(idx1, 3, idx2, 3);

  // Disjoint sets: distance = 1
  EXPECT_DOUBLE_EQ(dist, 1.0);
}

TEST(DistanceTest, JaccardDistancePartialOverlap) {
  matgen_index_t idx1[] = {0, 1, 2, 3};
  matgen_index_t idx2[] = {2, 3, 4, 5};

  matgen_value_t dist = matgen_sparse_jaccard_distance(idx1, 4, idx2, 4);

  // Intersection: {2, 3} = 2 elements
  // Union: {0, 1, 2, 3, 4, 5} = 6 elements
  // Jaccard similarity = 2/6 = 1/3
  // Jaccard distance = 1 - 1/3 = 2/3
  EXPECT_NEAR(dist, 2.0 / 3.0, EPSILON);
}

TEST(DistanceTest, JaccardDistanceOneEmpty) {
  matgen_index_t idx[] = {0, 1, 2};

  matgen_value_t dist1 = matgen_sparse_jaccard_distance(idx, 3, nullptr, 0);
  EXPECT_DOUBLE_EQ(dist1, 1.0);  // One empty, one non-empty: distance = 1

  matgen_value_t dist2 = matgen_sparse_jaccard_distance(nullptr, 0, idx, 3);
  EXPECT_DOUBLE_EQ(dist2, 1.0);
}

TEST(DistanceTest, JaccardDistanceBothEmpty) {
  matgen_value_t dist = matgen_sparse_jaccard_distance(nullptr, 0, nullptr, 0);
  EXPECT_DOUBLE_EQ(dist, 0.0);  // Both empty: distance = 0
}

TEST(DistanceTest, JaccardDistanceSingleElementOverlap) {
  matgen_index_t idx1[] = {5};
  matgen_index_t idx2[] = {5};

  matgen_value_t dist = matgen_sparse_jaccard_distance(idx1, 1, idx2, 1);
  EXPECT_DOUBLE_EQ(dist, 0.0);
}

TEST(DistanceTest, JaccardDistanceSingleElementNoOverlap) {
  matgen_index_t idx1[] = {3};
  matgen_index_t idx2[] = {7};

  matgen_value_t dist = matgen_sparse_jaccard_distance(idx1, 1, idx2, 1);
  EXPECT_DOUBLE_EQ(dist, 1.0);
}

// =============================================================================
// Cross-Metric Consistency Tests
// =============================================================================

TEST(DistanceTest, EuclideanVsManhattan) {
  // For vectors aligned to axes, Euclidean should equal Manhattan
  matgen_index_t idx1[] = {0};
  matgen_value_t val1[] = {3.0};

  matgen_index_t idx2[] = {0};
  matgen_value_t val2[] = {7.0};

  matgen_value_t euclidean =
      matgen_sparse_euclidean_distance(idx1, val1, 1, idx2, val2, 1);
  matgen_value_t manhattan =
      matgen_sparse_manhattan_distance(idx1, val1, 1, idx2, val2, 1);

  EXPECT_DOUBLE_EQ(euclidean, manhattan);
  EXPECT_DOUBLE_EQ(euclidean, 4.0);
}

TEST(DistanceTest, DistanceSymmetry) {
  matgen_index_t idx1[] = {0, 2, 4};
  matgen_value_t val1[] = {1.0, 2.0, 3.0};

  matgen_index_t idx2[] = {1, 3, 5};
  matgen_value_t val2[] = {4.0, 5.0, 6.0};

  // NOLINTBEGIN(readability-suspicious-call-argument)

  // Euclidean
  matgen_value_t euc1 =
      matgen_sparse_euclidean_distance(idx1, val1, 3, idx2, val2, 3);
  matgen_value_t euc2 =
      matgen_sparse_euclidean_distance(idx2, val2, 3, idx1, val1, 3);
  EXPECT_DOUBLE_EQ(euc1, euc2);

  // Manhattan
  matgen_value_t man1 =
      matgen_sparse_manhattan_distance(idx1, val1, 3, idx2, val2, 3);
  matgen_value_t man2 =
      matgen_sparse_manhattan_distance(idx2, val2, 3, idx1, val1, 3);
  EXPECT_DOUBLE_EQ(man1, man2);

  // Cosine
  matgen_value_t cos1 =
      matgen_sparse_cosine_distance(idx1, val1, 3, idx2, val2, 3);
  matgen_value_t cos2 =
      matgen_sparse_cosine_distance(idx2, val2, 3, idx1, val1, 3);
  EXPECT_NEAR(cos1, cos2, EPSILON);

  // Jaccard
  matgen_value_t jac1 = matgen_sparse_jaccard_distance(idx1, 3, idx2, 3);
  matgen_value_t jac2 = matgen_sparse_jaccard_distance(idx2, 3, idx1, 3);
  EXPECT_DOUBLE_EQ(jac1, jac2);

  // NOLINTEND(readability-suspicious-call-argument)
}
