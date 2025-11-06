#include <gtest/gtest.h>
#include <matgen/core/csr_matrix.h>
#include <matgen/core/types.h>

// Test fixture for CSR matrix
class CSRMatrixTest : public ::testing::Test {
 protected:
  matgen_csr_matrix_t* matrix;  // NOLINT

  void SetUp() override { matrix = nullptr; }

  void TearDown() override {
    if (matrix != nullptr) {
      matgen_csr_destroy(matrix);
      matrix = nullptr;
    }
  }

  // Helper: Create a simple 3x3 identity-like matrix
  // [1 0 0]
  // [0 2 0]
  // [0 0 3]
  static matgen_csr_matrix_t* create_identity_3x3() {
    matgen_csr_matrix_t* m = matgen_csr_create(3, 3, 3);
    if (m == nullptr) {
      return nullptr;
    }

    // row_ptr: [0, 1, 2, 3]
    m->row_ptr[0] = 0;
    m->row_ptr[1] = 1;
    m->row_ptr[2] = 2;
    m->row_ptr[3] = 3;

    // col_indices: [0, 1, 2]
    m->col_indices[0] = 0;
    m->col_indices[1] = 1;
    m->col_indices[2] = 2;

    // values: [1.0, 2.0, 3.0]
    m->values[0] = 1.0;
    m->values[1] = 2.0;
    m->values[2] = 3.0;

    return m;
  }
};

// =============================================================================
// Creation and Destruction Tests
// =============================================================================

TEST_F(CSRMatrixTest, CreateValidMatrix) {
  matrix = matgen_csr_create(10, 20, 50);

  ASSERT_NE(matrix, nullptr);
  EXPECT_EQ(matrix->rows, 10);
  EXPECT_EQ(matrix->cols, 20);
  EXPECT_EQ(matrix->nnz, 50);
  EXPECT_NE(matrix->row_ptr, nullptr);
  EXPECT_NE(matrix->col_indices, nullptr);
  EXPECT_NE(matrix->values, nullptr);

  // row_ptr should be initialized to zeros
  EXPECT_EQ(matrix->row_ptr[0], 0);
}

TEST_F(CSRMatrixTest, CreateEmptyMatrix) {
  matrix = matgen_csr_create(5, 5, 0);

  ASSERT_NE(matrix, nullptr);
  EXPECT_EQ(matrix->nnz, 0);
}

TEST_F(CSRMatrixTest, CreateInvalidDimensions) {
  matrix = matgen_csr_create(0, 10, 0);
  EXPECT_EQ(matrix, nullptr);

  matrix = matgen_csr_create(10, 0, 0);
  EXPECT_EQ(matrix, nullptr);
}

TEST_F(CSRMatrixTest, DestroyNullMatrix) {
  // Should not crash
  matgen_csr_destroy(nullptr);
}

// =============================================================================
// Get Entry Tests
// =============================================================================

TEST_F(CSRMatrixTest, GetExistingEntry) {
  matrix = create_identity_3x3();
  ASSERT_NE(matrix, nullptr);

  matgen_value_t value;
  EXPECT_EQ(matgen_csr_get(matrix, 0, 0, &value), MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, 1.0);

  EXPECT_EQ(matgen_csr_get(matrix, 1, 1, &value), MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, 2.0);

  EXPECT_EQ(matgen_csr_get(matrix, 2, 2, &value), MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, 3.0);
}

TEST_F(CSRMatrixTest, GetNonExistentEntry) {
  matrix = create_identity_3x3();
  ASSERT_NE(matrix, nullptr);

  matgen_value_t value;
  EXPECT_EQ(matgen_csr_get(matrix, 0, 1, &value),
            MATGEN_ERROR_INVALID_ARGUMENT);
  EXPECT_DOUBLE_EQ(value, 0.0);

  EXPECT_EQ(matgen_csr_get(matrix, 1, 0, &value),
            MATGEN_ERROR_INVALID_ARGUMENT);
  EXPECT_DOUBLE_EQ(value, 0.0);
}

TEST_F(CSRMatrixTest, GetOutOfBounds) {
  matrix = create_identity_3x3();
  ASSERT_NE(matrix, nullptr);

  matgen_value_t value;
  EXPECT_EQ(matgen_csr_get(matrix, 3, 0, &value),
            MATGEN_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(matgen_csr_get(matrix, 0, 3, &value),
            MATGEN_ERROR_INVALID_ARGUMENT);
}

TEST_F(CSRMatrixTest, GetWithNullValue) {
  matrix = create_identity_3x3();
  ASSERT_NE(matrix, nullptr);

  // Should not crash with NULL value pointer
  EXPECT_EQ(matgen_csr_get(matrix, 0, 0, nullptr), MATGEN_SUCCESS);
}

TEST_F(CSRMatrixTest, GetNullMatrix) {
  matgen_value_t value;
  EXPECT_EQ(matgen_csr_get(nullptr, 0, 0, &value),
            MATGEN_ERROR_INVALID_ARGUMENT);
}

TEST_F(CSRMatrixTest, HasEntryExists) {
  matrix = create_identity_3x3();
  ASSERT_NE(matrix, nullptr);

  EXPECT_TRUE(matgen_csr_has_entry(matrix, 0, 0));
  EXPECT_TRUE(matgen_csr_has_entry(matrix, 1, 1));
  EXPECT_TRUE(matgen_csr_has_entry(matrix, 2, 2));
}

TEST_F(CSRMatrixTest, HasEntryNotExists) {
  matrix = create_identity_3x3();
  ASSERT_NE(matrix, nullptr);

  EXPECT_FALSE(matgen_csr_has_entry(matrix, 0, 1));
  EXPECT_FALSE(matgen_csr_has_entry(matrix, 1, 0));
}

// =============================================================================
// Row Operations Tests
// =============================================================================

TEST_F(CSRMatrixTest, RowNnz) {
  matrix = create_identity_3x3();
  ASSERT_NE(matrix, nullptr);

  EXPECT_EQ(matgen_csr_row_nnz(matrix, 0), 1);
  EXPECT_EQ(matgen_csr_row_nnz(matrix, 1), 1);
  EXPECT_EQ(matgen_csr_row_nnz(matrix, 2), 1);
}

TEST_F(CSRMatrixTest, RowNnzEmptyRow) {
  matrix = matgen_csr_create(5, 5, 2);
  ASSERT_NE(matrix, nullptr);

  // Create matrix with empty middle rows
  matrix->row_ptr[0] = 0;
  matrix->row_ptr[1] = 1;
  matrix->row_ptr[2] = 1;  // Empty row
  matrix->row_ptr[3] = 1;  // Empty row
  matrix->row_ptr[4] = 2;
  matrix->row_ptr[5] = 2;

  matrix->col_indices[0] = 0;
  matrix->col_indices[1] = 4;
  matrix->values[0] = 1.0;
  matrix->values[1] = 2.0;

  EXPECT_EQ(matgen_csr_row_nnz(matrix, 0), 1);
  EXPECT_EQ(matgen_csr_row_nnz(matrix, 1), 0);  // Empty
  EXPECT_EQ(matgen_csr_row_nnz(matrix, 2), 0);  // Empty
  EXPECT_EQ(matgen_csr_row_nnz(matrix, 3), 1);
}

TEST_F(CSRMatrixTest, GetRowRange) {
  matrix = create_identity_3x3();
  ASSERT_NE(matrix, nullptr);

  matgen_size_t start;
  matgen_size_t end;

  EXPECT_EQ(matgen_csr_get_row_range(matrix, 0, &start, &end), MATGEN_SUCCESS);
  EXPECT_EQ(start, 0);
  EXPECT_EQ(end, 1);

  EXPECT_EQ(matgen_csr_get_row_range(matrix, 1, &start, &end), MATGEN_SUCCESS);
  EXPECT_EQ(start, 1);
  EXPECT_EQ(end, 2);

  EXPECT_EQ(matgen_csr_get_row_range(matrix, 2, &start, &end), MATGEN_SUCCESS);
  EXPECT_EQ(start, 2);
  EXPECT_EQ(end, 3);
}

TEST_F(CSRMatrixTest, GetRowRangeInvalid) {
  matrix = create_identity_3x3();
  ASSERT_NE(matrix, nullptr);

  matgen_size_t start;
  matgen_size_t end;

  // Out of bounds row
  EXPECT_EQ(matgen_csr_get_row_range(matrix, 3, &start, &end),
            MATGEN_ERROR_INVALID_ARGUMENT);

  // NULL pointers
  EXPECT_EQ(matgen_csr_get_row_range(matrix, 0, nullptr, &end),
            MATGEN_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(matgen_csr_get_row_range(matrix, 0, &start, nullptr),
            MATGEN_ERROR_INVALID_ARGUMENT);
}

// =============================================================================
// Validation Tests
// =============================================================================

TEST_F(CSRMatrixTest, ValidateValidMatrix) {
  matrix = create_identity_3x3();
  ASSERT_NE(matrix, nullptr);

  EXPECT_TRUE(matgen_csr_validate(matrix));
}

TEST_F(CSRMatrixTest, ValidateNullMatrix) {
  EXPECT_FALSE(matgen_csr_validate(nullptr));
}

TEST_F(CSRMatrixTest, ValidateInvalidRowPtr) {
  matrix = matgen_csr_create(3, 3, 3);
  ASSERT_NE(matrix, nullptr);

  // Invalid: row_ptr[0] != 0
  matrix->row_ptr[0] = 1;
  matrix->row_ptr[1] = 2;
  matrix->row_ptr[2] = 3;
  matrix->row_ptr[3] = 3;

  EXPECT_FALSE(matgen_csr_validate(matrix));
}

TEST_F(CSRMatrixTest, ValidateInvalidNnz) {
  matrix = create_identity_3x3();
  ASSERT_NE(matrix, nullptr);

  // Invalid: row_ptr[rows] != nnz
  matrix->row_ptr[3] = 5;  // Should be 3

  EXPECT_FALSE(matgen_csr_validate(matrix));
}

TEST_F(CSRMatrixTest, ValidateNonMonotonicRowPtr) {
  matrix = matgen_csr_create(3, 3, 3);
  ASSERT_NE(matrix, nullptr);

  // Invalid: non-monotonic
  matrix->row_ptr[0] = 0;
  matrix->row_ptr[1] = 2;
  matrix->row_ptr[2] = 1;  // Goes backwards!
  matrix->row_ptr[3] = 3;

  EXPECT_FALSE(matgen_csr_validate(matrix));
}

TEST_F(CSRMatrixTest, ValidateColumnOutOfBounds) {
  matrix = create_identity_3x3();
  ASSERT_NE(matrix, nullptr);

  // Invalid: column index out of bounds
  matrix->col_indices[1] = 5;  // Max should be 2

  EXPECT_FALSE(matgen_csr_validate(matrix));
}

TEST_F(CSRMatrixTest, ValidateUnsortedColumns) {
  matrix = matgen_csr_create(2, 5, 3);
  ASSERT_NE(matrix, nullptr);

  // Row 0 has 3 entries
  matrix->row_ptr[0] = 0;
  matrix->row_ptr[1] = 3;
  matrix->row_ptr[2] = 3;

  // Columns not sorted within row
  matrix->col_indices[0] = 2;
  matrix->col_indices[1] = 4;
  matrix->col_indices[2] = 1;  // Should come before 2!

  matrix->values[0] = 1.0;
  matrix->values[1] = 2.0;
  matrix->values[2] = 3.0;

  EXPECT_FALSE(matgen_csr_validate(matrix));
}

// =============================================================================
// Memory and Info Tests
// =============================================================================

TEST_F(CSRMatrixTest, MemoryUsage) {
  matrix = matgen_csr_create(5, 5, 10);
  ASSERT_NE(matrix, nullptr);

  matgen_size_t memory = matgen_csr_memory_usage(matrix);

  matgen_size_t expected = sizeof(matgen_csr_matrix_t) +
                           (6 * sizeof(matgen_size_t)) +    // row_ptr (rows+1)
                           (10 * sizeof(matgen_index_t)) +  // col_indices
                           (10 * sizeof(matgen_value_t));   // values

  EXPECT_EQ(memory, expected);
}

TEST_F(CSRMatrixTest, MemoryUsageNull) {
  EXPECT_EQ(matgen_csr_memory_usage(nullptr), 0);
}

TEST_F(CSRMatrixTest, PrintInfo) {
  matrix = create_identity_3x3();
  ASSERT_NE(matrix, nullptr);

  // Should not crash
  matgen_csr_print_info(matrix, stdout);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(CSRMatrixTest, LargeMatrix) {
  matrix = matgen_csr_create(1000000, 1000000, 1000);

  ASSERT_NE(matrix, nullptr);
  EXPECT_EQ(matrix->rows, 1000000);
  EXPECT_EQ(matrix->cols, 1000000);
}

TEST_F(CSRMatrixTest, NonSquareMatrix) {
  matrix = matgen_csr_create(10, 100, 50);

  ASSERT_NE(matrix, nullptr);
  EXPECT_EQ(matrix->rows, 10);
  EXPECT_EQ(matrix->cols, 100);
}
