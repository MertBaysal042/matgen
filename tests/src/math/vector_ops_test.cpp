#include <gtest/gtest.h>
#include <matgen/math/vector_ops.h>

#include <cmath>

const double EPSILON = 1e-10;

// =============================================================================
// Norms and Dot Product Tests
// =============================================================================

TEST(VectorOpsTest, DotProduct) {
  matgen_value_t x[] = {1.0, 2.0, 3.0, 4.0};
  matgen_value_t y[] = {2.0, 3.0, 4.0, 5.0};

  matgen_value_t result = matgen_vec_dot(x, y, 4);

  // Expected: 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
  EXPECT_DOUBLE_EQ(result, 40.0);
}

TEST(VectorOpsTest, DotProductZero) {
  matgen_value_t x[] = {1.0, 2.0, 3.0};
  matgen_value_t y[] = {0.0, 0.0, 0.0};

  matgen_value_t result = matgen_vec_dot(x, y, 3);
  EXPECT_DOUBLE_EQ(result, 0.0);
}

TEST(VectorOpsTest, DotProductOrthogonal) {
  matgen_value_t x[] = {1.0, 0.0};
  matgen_value_t y[] = {0.0, 1.0};

  matgen_value_t result = matgen_vec_dot(x, y, 2);
  EXPECT_DOUBLE_EQ(result, 0.0);
}

TEST(VectorOpsTest, Norm2) {
  matgen_value_t x[] = {3.0, 4.0};

  matgen_value_t norm = matgen_vec_norm2(x, 2);
  EXPECT_DOUBLE_EQ(norm, 5.0);  // sqrt(9 + 16) = 5
}

TEST(VectorOpsTest, Norm2Zero) {
  matgen_value_t x[] = {0.0, 0.0, 0.0};

  matgen_value_t norm = matgen_vec_norm2(x, 3);
  EXPECT_DOUBLE_EQ(norm, 0.0);
}

TEST(VectorOpsTest, Norm1) {
  matgen_value_t x[] = {1.0, -2.0, 3.0, -4.0};

  matgen_value_t norm = matgen_vec_norm1(x, 4);
  EXPECT_DOUBLE_EQ(norm, 10.0);  // 1 + 2 + 3 + 4 = 10
}

TEST(VectorOpsTest, NormInf) {
  matgen_value_t x[] = {1.0, -5.5, 3.0, 4.0};

  matgen_value_t norm = matgen_vec_norminf(x, 4);
  EXPECT_DOUBLE_EQ(norm, 5.5);
}

TEST(VectorOpsTest, NormInfAllZeros) {
  matgen_value_t x[] = {0.0, 0.0, 0.0};

  matgen_value_t norm = matgen_vec_norminf(x, 3);
  EXPECT_DOUBLE_EQ(norm, 0.0);
}

// =============================================================================
// Basic Vector Operations
// =============================================================================

TEST(VectorOpsTest, Scale) {
  matgen_value_t x[] = {1.0, 2.0, 3.0, 4.0};
  matgen_value_t y[4];

  matgen_vec_scale(2.5, x, y, 4);

  EXPECT_DOUBLE_EQ(y[0], 2.5);
  EXPECT_DOUBLE_EQ(y[1], 5.0);
  EXPECT_DOUBLE_EQ(y[2], 7.5);
  EXPECT_DOUBLE_EQ(y[3], 10.0);
}

TEST(VectorOpsTest, ScaleInPlace) {
  matgen_value_t x[] = {1.0, 2.0, 3.0};

  matgen_vec_scale(3.0, x, x, 3);  // In-place

  EXPECT_DOUBLE_EQ(x[0], 3.0);
  EXPECT_DOUBLE_EQ(x[1], 6.0);
  EXPECT_DOUBLE_EQ(x[2], 9.0);
}

TEST(VectorOpsTest, ScaleByZero) {
  matgen_value_t x[] = {1.0, 2.0, 3.0};
  matgen_value_t y[3];

  matgen_vec_scale(0.0, x, y, 3);

  EXPECT_DOUBLE_EQ(y[0], 0.0);
  EXPECT_DOUBLE_EQ(y[1], 0.0);
  EXPECT_DOUBLE_EQ(y[2], 0.0);
}

TEST(VectorOpsTest, Add) {
  matgen_value_t x[] = {1.0, 2.0, 3.0};
  matgen_value_t y[] = {4.0, 5.0, 6.0};
  matgen_value_t z[3];

  matgen_vec_add(x, y, z, 3);

  EXPECT_DOUBLE_EQ(z[0], 5.0);
  EXPECT_DOUBLE_EQ(z[1], 7.0);
  EXPECT_DOUBLE_EQ(z[2], 9.0);
}

TEST(VectorOpsTest, AddInPlace) {
  matgen_value_t x[] = {1.0, 2.0, 3.0};
  matgen_value_t y[] = {4.0, 5.0, 6.0};

  matgen_vec_add(x, y, x, 3);  // x = x + y

  EXPECT_DOUBLE_EQ(x[0], 5.0);
  EXPECT_DOUBLE_EQ(x[1], 7.0);
  EXPECT_DOUBLE_EQ(x[2], 9.0);
}

TEST(VectorOpsTest, Subtract) {
  matgen_value_t x[] = {5.0, 7.0, 9.0};
  matgen_value_t y[] = {1.0, 2.0, 3.0};
  matgen_value_t z[3];

  matgen_vec_sub(x, y, z, 3);

  EXPECT_DOUBLE_EQ(z[0], 4.0);
  EXPECT_DOUBLE_EQ(z[1], 5.0);
  EXPECT_DOUBLE_EQ(z[2], 6.0);
}

TEST(VectorOpsTest, SubtractInPlace) {
  matgen_value_t x[] = {5.0, 7.0, 9.0};
  matgen_value_t y[] = {1.0, 2.0, 3.0};

  matgen_vec_sub(x, y, x, 3);  // x = x - y

  EXPECT_DOUBLE_EQ(x[0], 4.0);
  EXPECT_DOUBLE_EQ(x[1], 5.0);
  EXPECT_DOUBLE_EQ(x[2], 6.0);
}

TEST(VectorOpsTest, AXPY) {
  matgen_value_t x[] = {1.0, 2.0, 3.0};
  matgen_value_t y[] = {4.0, 5.0, 6.0};

  matgen_vec_axpy(2.0, x, y, 3);  // y = 2*x + y

  EXPECT_DOUBLE_EQ(y[0], 6.0);   // 2*1 + 4 = 6
  EXPECT_DOUBLE_EQ(y[1], 9.0);   // 2*2 + 5 = 9
  EXPECT_DOUBLE_EQ(y[2], 12.0);  // 2*3 + 6 = 12
}

TEST(VectorOpsTest, AXPYZeroAlpha) {
  matgen_value_t x[] = {1.0, 2.0, 3.0};
  matgen_value_t y[] = {4.0, 5.0, 6.0};
  matgen_value_t y_original[] = {4.0, 5.0, 6.0};

  matgen_vec_axpy(0.0, x, y, 3);  // y should remain unchanged

  EXPECT_DOUBLE_EQ(y[0], y_original[0]);
  EXPECT_DOUBLE_EQ(y[1], y_original[1]);
  EXPECT_DOUBLE_EQ(y[2], y_original[2]);
}

TEST(VectorOpsTest, AXPBY) {
  matgen_value_t x[] = {1.0, 2.0, 3.0};
  matgen_value_t y[] = {4.0, 5.0, 6.0};
  matgen_value_t z[3];

  matgen_vec_axpby(2.0, x, 3.0, y, z, 3);  // z = 2*x + 3*y

  EXPECT_DOUBLE_EQ(z[0], 14.0);  // 2*1 + 3*4 = 14
  EXPECT_DOUBLE_EQ(z[1], 19.0);  // 2*2 + 3*5 = 19
  EXPECT_DOUBLE_EQ(z[2], 24.0);  // 2*3 + 3*6 = 24
}

TEST(VectorOpsTest, AXPBYInPlace) {
  matgen_value_t x[] = {1.0, 2.0, 3.0};
  matgen_value_t y[] = {4.0, 5.0, 6.0};

  matgen_vec_axpby(2.0, x, 3.0, y, x, 3);  // x = 2*x + 3*y

  EXPECT_DOUBLE_EQ(x[0], 14.0);
  EXPECT_DOUBLE_EQ(x[1], 19.0);
  EXPECT_DOUBLE_EQ(x[2], 24.0);
}

// =============================================================================
// Utility Operations
// =============================================================================

TEST(VectorOpsTest, Copy) {
  matgen_value_t x[] = {1.5, 2.5, 3.5, 4.5};
  matgen_value_t y[4];

  matgen_vec_copy(x, y, 4);

  for (int i = 0; i < 4; i++) {
    EXPECT_DOUBLE_EQ(y[i], x[i]);
  }
}

TEST(VectorOpsTest, Zero) {
  matgen_value_t x[] = {1.0, 2.0, 3.0, 4.0};

  matgen_vec_zero(x, 4);

  EXPECT_DOUBLE_EQ(x[0], 0.0);
  EXPECT_DOUBLE_EQ(x[1], 0.0);
  EXPECT_DOUBLE_EQ(x[2], 0.0);
  EXPECT_DOUBLE_EQ(x[3], 0.0);
}

TEST(VectorOpsTest, Fill) {
  matgen_value_t x[5];

  matgen_vec_fill(x, 7.5, 5);

  for (int i = 0; i < 5; i++) {
    EXPECT_DOUBLE_EQ(x[i], 7.5);
  }
}

TEST(VectorOpsTest, FillZero) {
  matgen_value_t x[3];

  matgen_vec_fill(x, 0.0, 3);

  EXPECT_DOUBLE_EQ(x[0], 0.0);
  EXPECT_DOUBLE_EQ(x[1], 0.0);
  EXPECT_DOUBLE_EQ(x[2], 0.0);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(VectorOpsTest, EmptyVectors) {
  matgen_value_t* x = nullptr;
  matgen_value_t* y = nullptr;

  // Operations on size 0 should not crash
  matgen_value_t dot = matgen_vec_dot(x, y, 0);
  EXPECT_DOUBLE_EQ(dot, 0.0);

  matgen_value_t norm = matgen_vec_norm2(x, 0);
  EXPECT_DOUBLE_EQ(norm, 0.0);
}

TEST(VectorOpsTest, SingleElement) {
  matgen_value_t x[] = {5.0};
  matgen_value_t y[] = {3.0};

  matgen_value_t dot = matgen_vec_dot(x, y, 1);
  EXPECT_DOUBLE_EQ(dot, 15.0);

  matgen_value_t norm = matgen_vec_norm2(x, 1);
  EXPECT_DOUBLE_EQ(norm, 5.0);
}

TEST(VectorOpsTest, LargeVector) {
  const size_t n = 10000;
  matgen_value_t* x = new matgen_value_t[n];
  matgen_value_t* y = new matgen_value_t[n];

  // Fill with pattern
  for (size_t i = 0; i < n; i++) {
    x[i] = static_cast<matgen_value_t>(i);
    y[i] = static_cast<matgen_value_t>(i * 2);
  }

  // Test some operations
  matgen_vec_axpy(1.5, x, y, n);

  // Verify a few elements
  EXPECT_DOUBLE_EQ(y[0], 0.0);      // 1.5*0 + 0 = 0
  EXPECT_DOUBLE_EQ(y[1], 3.5);      // 1.5*1 + 2 = 3.5
  EXPECT_DOUBLE_EQ(y[100], 350.0);  // 1.5*100 + 200 = 350

  delete[] x;
  delete[] y;
}

// =============================================================================
// Numerical Stability Tests
// =============================================================================

TEST(VectorOpsTest, NormWithSmallValues) {
  matgen_value_t x[] = {1e-10, 1e-10, 1e-10};

  matgen_value_t norm = matgen_vec_norm2(x, 3);
  matgen_value_t expected = sqrt(3.0) * 1e-10;

  EXPECT_NEAR(norm, expected, 1e-20);
}

TEST(VectorOpsTest, DotProductWithLargeAndSmall) {
  matgen_value_t x[] = {1e10, 1e-10};
  matgen_value_t y[] = {1e-10, 1e10};

  matgen_value_t result = matgen_vec_dot(x, y, 2);

  // Expected: 1e10*1e-10 + 1e-10*1e10 = 1 + 1 = 2
  EXPECT_NEAR(result, 2.0, 1e-6);
}
