#include <gtest/gtest.h>
#include <matgen/core/coo_matrix.h>
#include <matgen/core/types.h>

// Test fixture for COO matrix
class COOMatrixTest : public ::testing::Test {
 protected:
  matgen_coo_matrix_t* matrix;  // NOLINT

  void SetUp() override { matrix = nullptr; }

  void TearDown() override {
    if (matrix != nullptr) {
      matgen_coo_destroy(matrix);
      matrix = nullptr;
    }
  }
};

// =============================================================================
// Creation and Destruction Tests
// =============================================================================

TEST_F(COOMatrixTest, CreateValidMatrix) {
  matrix = matgen_coo_create(10, 20, 50);

  ASSERT_NE(matrix, nullptr);
  EXPECT_EQ(matrix->rows, 10);
  EXPECT_EQ(matrix->cols, 20);
  EXPECT_EQ(matrix->nnz, 0);
  EXPECT_GE(matrix->capacity, 50);
  EXPECT_TRUE(matrix->is_sorted);
  EXPECT_NE(matrix->row_indices, nullptr);
  EXPECT_NE(matrix->col_indices, nullptr);
  EXPECT_NE(matrix->values, nullptr);
}

TEST_F(COOMatrixTest, CreateWithZeroHint) {
  matrix = matgen_coo_create(5, 5, 0);

  ASSERT_NE(matrix, nullptr);
  EXPECT_GT(matrix->capacity, 0);  // Should use default capacity
}

TEST_F(COOMatrixTest, CreateInvalidDimensions) {
  matrix = matgen_coo_create(0, 10, 0);
  EXPECT_EQ(matrix, nullptr);

  matrix = matgen_coo_create(10, 0, 0);
  EXPECT_EQ(matrix, nullptr);
}

TEST_F(COOMatrixTest, DestroyNullMatrix) {
  // Should not crash
  matgen_coo_destroy(nullptr);
}

// =============================================================================
// Add Entry Tests
// =============================================================================

TEST_F(COOMatrixTest, AddSingleEntry) {
  matrix = matgen_coo_create(5, 5, 10);

  matgen_error_t err = matgen_coo_add_entry(matrix, 2, 3, 4.5);

  EXPECT_EQ(err, MATGEN_SUCCESS);
  EXPECT_EQ(matrix->nnz, 1);
  EXPECT_EQ(matrix->row_indices[0], 2);
  EXPECT_EQ(matrix->col_indices[0], 3);
  EXPECT_DOUBLE_EQ(matrix->values[0], 4.5);
  EXPECT_FALSE(matrix->is_sorted);  // No longer sorted after adding
}

TEST_F(COOMatrixTest, AddMultipleEntries) {
  matrix = matgen_coo_create(5, 5, 10);

  EXPECT_EQ(matgen_coo_add_entry(matrix, 0, 0, 1.0), MATGEN_SUCCESS);
  EXPECT_EQ(matgen_coo_add_entry(matrix, 1, 2, 2.0), MATGEN_SUCCESS);
  EXPECT_EQ(matgen_coo_add_entry(matrix, 3, 4, 3.0), MATGEN_SUCCESS);

  EXPECT_EQ(matrix->nnz, 3);
}

TEST_F(COOMatrixTest, AddEntryOutOfBounds) {
  matrix = matgen_coo_create(5, 5, 10);

  EXPECT_EQ(matgen_coo_add_entry(matrix, 5, 2, 1.0),
            MATGEN_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(matgen_coo_add_entry(matrix, 2, 5, 1.0),
            MATGEN_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(matrix->nnz, 0);  // No entries added
}

TEST_F(COOMatrixTest, AddEntryNullMatrix) {
  EXPECT_EQ(matgen_coo_add_entry(nullptr, 0, 0, 1.0),
            MATGEN_ERROR_INVALID_ARGUMENT);
}

TEST_F(COOMatrixTest, AddEntryWithResize) {
  matrix = matgen_coo_create(10, 10, 2);  // Small initial capacity

  // Add more than initial capacity
  for (matgen_index_t i = 0; i < 10; i++) {
    EXPECT_EQ(matgen_coo_add_entry(matrix, i, i, (matgen_value_t)i),
              MATGEN_SUCCESS);
  }

  EXPECT_EQ(matrix->nnz, 10);
  EXPECT_GE(matrix->capacity, 10);  // Should have grown
}

// =============================================================================
// Sorting Tests
// =============================================================================

TEST_F(COOMatrixTest, SortEmptyMatrix) {
  matrix = matgen_coo_create(5, 5, 10);

  EXPECT_EQ(matgen_coo_sort(matrix), MATGEN_SUCCESS);
  EXPECT_TRUE(matrix->is_sorted);
}

TEST_F(COOMatrixTest, SortSingleEntry) {
  matrix = matgen_coo_create(5, 5, 10);
  matgen_coo_add_entry(matrix, 2, 3, 1.0);

  EXPECT_EQ(matgen_coo_sort(matrix), MATGEN_SUCCESS);
  EXPECT_TRUE(matrix->is_sorted);
}

TEST_F(COOMatrixTest, SortMultipleEntries) {
  matrix = matgen_coo_create(5, 5, 10);

  // Add in unsorted order
  matgen_coo_add_entry(matrix, 3, 2, 3.0);
  matgen_coo_add_entry(matrix, 1, 1, 1.0);
  matgen_coo_add_entry(matrix, 2, 4, 2.0);

  EXPECT_EQ(matgen_coo_sort(matrix), MATGEN_SUCCESS);
  EXPECT_TRUE(matrix->is_sorted);

  // Check sorted order (row-major)
  EXPECT_EQ(matrix->row_indices[0], 1);
  EXPECT_EQ(matrix->col_indices[0], 1);

  EXPECT_EQ(matrix->row_indices[1], 2);
  EXPECT_EQ(matrix->col_indices[1], 4);

  EXPECT_EQ(matrix->row_indices[2], 3);
  EXPECT_EQ(matrix->col_indices[2], 2);
}

TEST_F(COOMatrixTest, SortNullMatrix) {
  EXPECT_EQ(matgen_coo_sort(nullptr), MATGEN_ERROR_INVALID_ARGUMENT);
}

// =============================================================================
// Get Entry Tests
// =============================================================================

TEST_F(COOMatrixTest, GetExistingEntry) {
  matrix = matgen_coo_create(5, 5, 10);
  matgen_coo_add_entry(matrix, 2, 3, 4.5);

  matgen_value_t value;
  EXPECT_EQ(matgen_coo_get(matrix, 2, 3, &value), MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, 4.5);
}

TEST_F(COOMatrixTest, GetNonExistentEntry) {
  matrix = matgen_coo_create(5, 5, 10);
  matgen_coo_add_entry(matrix, 2, 3, 4.5);

  matgen_value_t value;
  EXPECT_EQ(matgen_coo_get(matrix, 1, 1, &value),
            MATGEN_ERROR_INVALID_ARGUMENT);
  EXPECT_DOUBLE_EQ(value, 0.0);
}

TEST_F(COOMatrixTest, GetOutOfBounds) {
  matrix = matgen_coo_create(5, 5, 10);

  matgen_value_t value;
  EXPECT_EQ(matgen_coo_get(matrix, 5, 2, &value),
            MATGEN_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(matgen_coo_get(matrix, 2, 5, &value),
            MATGEN_ERROR_INVALID_ARGUMENT);
}

TEST_F(COOMatrixTest, GetWithNullValue) {
  matrix = matgen_coo_create(5, 5, 10);
  matgen_coo_add_entry(matrix, 2, 3, 4.5);

  // Should not crash with NULL value pointer
  EXPECT_EQ(matgen_coo_get(matrix, 2, 3, nullptr), MATGEN_SUCCESS);
}

TEST_F(COOMatrixTest, HasEntryExists) {
  matrix = matgen_coo_create(5, 5, 10);
  matgen_coo_add_entry(matrix, 2, 3, 4.5);

  EXPECT_TRUE(matgen_coo_has_entry(matrix, 2, 3));
}

TEST_F(COOMatrixTest, HasEntryNotExists) {
  matrix = matgen_coo_create(5, 5, 10);
  matgen_coo_add_entry(matrix, 2, 3, 4.5);

  EXPECT_FALSE(matgen_coo_has_entry(matrix, 1, 1));
}

// =============================================================================
// Reserve and Clear Tests
// =============================================================================

TEST_F(COOMatrixTest, ReserveCapacity) {
  matrix = matgen_coo_create(5, 5, 10);

  EXPECT_EQ(matgen_coo_reserve(matrix, 100), MATGEN_SUCCESS);
  EXPECT_GE(matrix->capacity, 100);
}

TEST_F(COOMatrixTest, ReserveSmallerCapacity) {
  matrix = matgen_coo_create(5, 5, 100);
  matgen_size_t old_capacity = matrix->capacity;

  // Reserving smaller capacity should be no-op
  EXPECT_EQ(matgen_coo_reserve(matrix, 50), MATGEN_SUCCESS);
  EXPECT_EQ(matrix->capacity, old_capacity);
}

TEST_F(COOMatrixTest, ClearMatrix) {
  matrix = matgen_coo_create(5, 5, 10);
  matgen_coo_add_entry(matrix, 0, 0, 1.0);
  matgen_coo_add_entry(matrix, 1, 1, 2.0);

  matgen_coo_clear(matrix);

  EXPECT_EQ(matrix->nnz, 0);
  EXPECT_TRUE(matrix->is_sorted);
}

TEST_F(COOMatrixTest, ClearNullMatrix) {
  // Should not crash
  matgen_coo_clear(nullptr);
}

// =============================================================================
// Validation Tests
// =============================================================================

TEST_F(COOMatrixTest, ValidateValidMatrix) {
  matrix = matgen_coo_create(5, 5, 10);
  matgen_coo_add_entry(matrix, 0, 0, 1.0);
  matgen_coo_add_entry(matrix, 2, 3, 2.0);

  EXPECT_TRUE(matgen_coo_validate(matrix));
}

TEST_F(COOMatrixTest, ValidateNullMatrix) {
  EXPECT_FALSE(matgen_coo_validate(nullptr));
}

TEST_F(COOMatrixTest, ValidateEmptyMatrix) {
  matrix = matgen_coo_create(5, 5, 10);
  EXPECT_TRUE(matgen_coo_validate(matrix));
}

// =============================================================================
// Memory and Info Tests
// =============================================================================

TEST_F(COOMatrixTest, MemoryUsage) {
  matrix = matgen_coo_create(5, 5, 100);

  matgen_size_t memory = matgen_coo_memory_usage(matrix);

  // Should include struct + arrays
  matgen_size_t expected = sizeof(matgen_coo_matrix_t) +
                           (100 * sizeof(matgen_index_t) * 2) +
                           (100 * sizeof(matgen_value_t));

  EXPECT_EQ(memory, expected);
}

TEST_F(COOMatrixTest, MemoryUsageNull) {
  EXPECT_EQ(matgen_coo_memory_usage(nullptr), 0);
}

TEST_F(COOMatrixTest, PrintInfo) {
  matrix = matgen_coo_create(10, 20, 50);
  matgen_coo_add_entry(matrix, 0, 0, 1.0);

  // Should not crash
  matgen_coo_print_info(matrix, stdout);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(COOMatrixTest, LargeMatrix) {
  matrix = matgen_coo_create(1000000, 1000000, 100);

  ASSERT_NE(matrix, nullptr);
  EXPECT_EQ(matrix->rows, 1000000);
  EXPECT_EQ(matrix->cols, 1000000);
}

TEST_F(COOMatrixTest, NonSquareMatrix) {
  matrix = matgen_coo_create(10, 100, 50);

  ASSERT_NE(matrix, nullptr);
  EXPECT_EQ(matrix->rows, 10);
  EXPECT_EQ(matrix->cols, 100);

  EXPECT_EQ(matgen_coo_add_entry(matrix, 5, 50, 1.0), MATGEN_SUCCESS);
}

TEST_F(COOMatrixTest, DuplicateEntries) {
  matrix = matgen_coo_create(5, 5, 10);

  // COO allows duplicate entries
  EXPECT_EQ(matgen_coo_add_entry(matrix, 2, 2, 1.0), MATGEN_SUCCESS);
  EXPECT_EQ(matgen_coo_add_entry(matrix, 2, 2, 2.0), MATGEN_SUCCESS);

  EXPECT_EQ(matrix->nnz, 2);
}
