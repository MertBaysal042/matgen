#include <gtest/gtest.h>
#include <matgen/core/coo_matrix.h>
#include <matgen/core/types.h>
#include <matgen/generators/random.h>

// Test fixture for random generation
class RandomGeneratorTest : public ::testing::Test {
 protected:
  matgen_coo_matrix_t* matrix;    // NOLINT
  matgen_random_config_t config;  // NOLINT

  void SetUp() override { matrix = nullptr; }

  void TearDown() override {
    if (matrix != nullptr) {
      matgen_coo_destroy(matrix);
      matrix = nullptr;
    }
  }
};

// =============================================================================
// Config Initialization Tests
// =============================================================================

TEST_F(RandomGeneratorTest, ConfigInit) {
  matgen_random_config_init(&config, 10, 20, 50);

  EXPECT_EQ(config.rows, 10);
  EXPECT_EQ(config.cols, 20);
  EXPECT_EQ(config.nnz, 50);
  EXPECT_DOUBLE_EQ(config.density, 0.0);

  EXPECT_EQ(config.distribution, MATGEN_DIST_UNIFORM);
  EXPECT_DOUBLE_EQ(config.min_value, 0.0);
  EXPECT_DOUBLE_EQ(config.max_value, 1.0);
  EXPECT_DOUBLE_EQ(config.mean, 0.0);
  EXPECT_DOUBLE_EQ(config.stddev, 1.0);
  EXPECT_DOUBLE_EQ(config.constant_value, 1.0);

  EXPECT_EQ(config.seed, 0);
  EXPECT_FALSE(config.allow_duplicates);
  EXPECT_TRUE(config.sorted);
}

// =============================================================================
// Random Generation Tests
// =============================================================================

TEST_F(RandomGeneratorTest, GenerateUniform) {
  matgen_random_config_init(&config, 10, 10, 20);
  config.seed = 42;  // Fixed seed for reproducibility

  matrix = matgen_random_generate(&config);
  ASSERT_NE(matrix, nullptr);

  EXPECT_EQ(matrix->rows, 10);
  EXPECT_EQ(matrix->cols, 10);
  EXPECT_EQ(matrix->nnz, 20);
  EXPECT_TRUE(matrix->is_sorted);

  // Check values are in [0, 1]
  for (matgen_size_t i = 0; i < matrix->nnz; i++) {
    EXPECT_GE(matrix->values[i], 0.0);
    EXPECT_LE(matrix->values[i], 1.0);
  }
}

TEST_F(RandomGeneratorTest, GenerateWithDensity) {
  matgen_random_config_init(&config, 10, 10, 0);
  config.density = 0.5;  // 50% density
  config.seed = 42;

  matrix = matgen_random_generate(&config);
  ASSERT_NE(matrix, nullptr);

  // Should generate ~50 entries (50% of 100)
  EXPECT_EQ(matrix->nnz, 50);
}

TEST_F(RandomGeneratorTest, GenerateConstant) {
  matgen_random_config_init(&config, 5, 5, 10);
  config.distribution = MATGEN_DIST_CONSTANT;
  config.constant_value = 3.14;
  config.seed = 42;

  matrix = matgen_random_generate(&config);
  ASSERT_NE(matrix, nullptr);

  // All values should be the constant
  for (matgen_size_t i = 0; i < matrix->nnz; i++) {
    EXPECT_DOUBLE_EQ(matrix->values[i], 3.14);
  }
}

TEST_F(RandomGeneratorTest, GenerateWithDuplicates) {
  matgen_random_config_init(&config, 5, 5, 30);
  config.allow_duplicates = true;
  config.seed = 42;

  matrix = matgen_random_generate(&config);
  ASSERT_NE(matrix, nullptr);

  EXPECT_EQ(matrix->nnz, 30);
  // Note: May contain duplicate (row,col) pairs
}

TEST_F(RandomGeneratorTest, GenerateNoDuplicates) {
  matgen_random_config_init(&config, 10, 10, 20);
  config.allow_duplicates = false;
  config.seed = 42;

  matrix = matgen_random_generate(&config);
  ASSERT_NE(matrix, nullptr);

  EXPECT_EQ(matrix->nnz, 20);

  // Sort and check for duplicates
  matgen_coo_sort(matrix);
  for (matgen_size_t i = 1; i < matrix->nnz; i++) {
    bool is_duplicate =
        (matrix->row_indices[i] == matrix->row_indices[i - 1]) &&
        (matrix->col_indices[i] == matrix->col_indices[i - 1]);
    EXPECT_FALSE(is_duplicate);
  }
}

TEST_F(RandomGeneratorTest, GenerateEmptyMatrix) {
  matgen_random_config_init(&config, 10, 10, 0);
  config.seed = 42;

  matrix = matgen_random_generate(&config);
  ASSERT_NE(matrix, nullptr);

  EXPECT_EQ(matrix->nnz, 0);
}

TEST_F(RandomGeneratorTest, GenerateInvalidDimensions) {
  matgen_random_config_init(&config, 0, 10, 10);

  matrix = matgen_random_generate(&config);
  EXPECT_EQ(matrix, nullptr);
}

TEST_F(RandomGeneratorTest, GenerateTooManyNoDuplicates) {
  matgen_random_config_init(&config, 3, 3, 100);  // Only 9 possible
  config.allow_duplicates = false;
  config.seed = 42;

  matrix = matgen_random_generate(&config);
  EXPECT_EQ(matrix, nullptr);  // Should fail
}

TEST_F(RandomGeneratorTest, GenerateUnsorted) {
  matgen_random_config_init(&config, 10, 10, 20);
  config.sorted = false;
  config.seed = 42;

  matrix = matgen_random_generate(&config);
  ASSERT_NE(matrix, nullptr);

  EXPECT_FALSE(matrix->is_sorted);
}

TEST_F(RandomGeneratorTest, ReproducibilityWithSeed) {
  // Generate two matrices with same seed
  matgen_random_config_init(&config, 10, 10, 20);
  config.seed = 12345;

  matrix = matgen_random_generate(&config);
  ASSERT_NE(matrix, nullptr);

  matgen_coo_matrix_t* matrix2 = matgen_random_generate(&config);
  ASSERT_NE(matrix2, nullptr);

  // Should be identical
  EXPECT_EQ(matrix->nnz, matrix2->nnz);
  for (matgen_size_t i = 0; i < matrix->nnz; i++) {
    EXPECT_EQ(matrix->row_indices[i], matrix2->row_indices[i]);
    EXPECT_EQ(matrix->col_indices[i], matrix2->col_indices[i]);
    EXPECT_DOUBLE_EQ(matrix->values[i], matrix2->values[i]);
  }

  matgen_coo_destroy(matrix2);
}

// =============================================================================
// Diagonal Matrix Tests
// =============================================================================

TEST_F(RandomGeneratorTest, DiagonalSquare) {
  matrix = matgen_random_diagonal(5, 5, MATGEN_DIST_UNIFORM, 0.0, 1.0, 42);
  ASSERT_NE(matrix, nullptr);

  EXPECT_EQ(matrix->rows, 5);
  EXPECT_EQ(matrix->cols, 5);
  EXPECT_EQ(matrix->nnz, 5);
  EXPECT_TRUE(matrix->is_sorted);

  // Check it's diagonal
  for (matgen_size_t i = 0; i < matrix->nnz; i++) {
    EXPECT_EQ(matrix->row_indices[i], matrix->col_indices[i]);
  }
}

TEST_F(RandomGeneratorTest, DiagonalNonSquare) {
  matrix = matgen_random_diagonal(3, 7, MATGEN_DIST_UNIFORM, 0.0, 1.0, 42);
  ASSERT_NE(matrix, nullptr);

  EXPECT_EQ(matrix->rows, 3);
  EXPECT_EQ(matrix->cols, 7);
  EXPECT_EQ(matrix->nnz, 3);  // min(3, 7) = 3

  // Check it's diagonal
  for (matgen_size_t i = 0; i < matrix->nnz; i++) {
    EXPECT_EQ(matrix->row_indices[i], matrix->col_indices[i]);
  }
}

TEST_F(RandomGeneratorTest, DiagonalConstant) {
  matrix = matgen_random_diagonal(4, 4, MATGEN_DIST_CONSTANT, 5.0, 0.0, 42);
  ASSERT_NE(matrix, nullptr);

  // All diagonal values should be 5.0
  for (matgen_size_t i = 0; i < matrix->nnz; i++) {
    EXPECT_DOUBLE_EQ(matrix->values[i], 5.0);
  }
}

// =============================================================================
// Tridiagonal Matrix Tests
// =============================================================================

TEST_F(RandomGeneratorTest, Tridiagonal) {
  matrix = matgen_random_tridiagonal(5, MATGEN_DIST_UNIFORM, 0.0, 1.0, 42);
  ASSERT_NE(matrix, nullptr);

  EXPECT_EQ(matrix->rows, 5);
  EXPECT_EQ(matrix->cols, 5);
  EXPECT_EQ(matrix->nnz, 13);  // 3*5 - 2 = 13
  EXPECT_TRUE(matrix->is_sorted);
}

TEST_F(RandomGeneratorTest, TridiagonalSize1) {
  matrix = matgen_random_tridiagonal(1, MATGEN_DIST_UNIFORM, 0.0, 1.0, 42);
  ASSERT_NE(matrix, nullptr);

  EXPECT_EQ(matrix->nnz, 1);  // Just the single diagonal element
}

TEST_F(RandomGeneratorTest, TridiagonalStructure) {
  matrix = matgen_random_tridiagonal(4, MATGEN_DIST_CONSTANT, 1.0, 0.0, 42);
  ASSERT_NE(matrix, nullptr);

  // Check structure: should have entries on main, upper, and lower diagonals
  EXPECT_EQ(matrix->nnz, 10);  // 3*4 - 2 = 10

  matgen_coo_sort(matrix);

  // Verify positions
  bool has_main_diag = true;
  bool has_upper_diag = true;
  bool has_lower_diag = true;

  for (matgen_index_t i = 0; i < 4; i++) {
    // Main diagonal
    if (!matgen_coo_has_entry(matrix, i, i)) {
      has_main_diag = false;
    }
    // Upper diagonal (except last row)
    if (i < 3 && !matgen_coo_has_entry(matrix, i, i + 1)) {
      has_upper_diag = false;
    }
    // Lower diagonal (except first row)
    if (i > 0 && !matgen_coo_has_entry(matrix, i, i - 1)) {
      has_lower_diag = false;
    }
  }

  EXPECT_TRUE(has_main_diag);
  EXPECT_TRUE(has_upper_diag);
  EXPECT_TRUE(has_lower_diag);
}

TEST_F(RandomGeneratorTest, TridiagonalInvalidSize) {
  matrix = matgen_random_tridiagonal(0, MATGEN_DIST_UNIFORM, 0.0, 1.0, 42);
  EXPECT_EQ(matrix, nullptr);
}

// =============================================================================
// Helper Function Tests
// =============================================================================

TEST_F(RandomGeneratorTest, RandomUniform) {
  matrix = matgen_random_uniform(10, 10, 20, 42);
  ASSERT_NE(matrix, nullptr);

  EXPECT_EQ(matrix->rows, 10);
  EXPECT_EQ(matrix->cols, 10);
  EXPECT_EQ(matrix->nnz, 20);

  // Values should be in [0, 1]
  for (matgen_size_t i = 0; i < matrix->nnz; i++) {
    EXPECT_GE(matrix->values[i], 0.0);
    EXPECT_LE(matrix->values[i], 1.0);
  }
}
