#include <gtest/gtest.h>
#include <matgen/core/conversion.h>
#include <matgen/core/coo_matrix.h>
#include <matgen/core/csr_matrix.h>
#include <matgen/core/types.h>

// Test fixture for conversion
class ConversionTest : public ::testing::Test {
 protected:
  matgen_coo_matrix_t* coo;  // NOLINT
  matgen_csr_matrix_t* csr;  // NOLINT

  void SetUp() override {
    coo = nullptr;
    csr = nullptr;
  }

  void TearDown() override {
    if (coo != nullptr) {
      matgen_coo_destroy(coo);
      coo = nullptr;
    }

    if (csr != nullptr) {
      matgen_csr_destroy(csr);
      csr = nullptr;
    }
  }

  // Helper: Create a simple 3x3 matrix
  // [1 0 2]
  // [0 3 0]
  // [4 0 5]
  static matgen_coo_matrix_t* create_test_coo() {
    matgen_coo_matrix_t* m = matgen_coo_create(3, 3, 5);
    if (m == nullptr) {
      return nullptr;
    }

    matgen_coo_add_entry(m, 0, 0, 1.0);
    matgen_coo_add_entry(m, 0, 2, 2.0);
    matgen_coo_add_entry(m, 1, 1, 3.0);
    matgen_coo_add_entry(m, 2, 0, 4.0);
    matgen_coo_add_entry(m, 2, 2, 5.0);

    return m;
  }
};

// =============================================================================
// COO to CSR Tests
// =============================================================================

TEST_F(ConversionTest, COOtoCSR_ValidMatrix) {
  coo = create_test_coo();
  ASSERT_NE(coo, nullptr);

  csr = matgen_coo_to_csr(coo);
  ASSERT_NE(csr, nullptr);

  // Check dimensions
  EXPECT_EQ(csr->rows, 3);
  EXPECT_EQ(csr->cols, 3);
  EXPECT_EQ(csr->nnz, 5);

  // Validate structure
  EXPECT_TRUE(matgen_csr_validate(csr));
}

TEST_F(ConversionTest, COOtoCSR_CheckValues) {
  coo = create_test_coo();
  ASSERT_NE(coo, nullptr);
  matgen_coo_sort(coo);

  csr = matgen_coo_to_csr(coo);
  ASSERT_NE(csr, nullptr);

  // Check row_ptr
  EXPECT_EQ(csr->row_ptr[0], 0);
  EXPECT_EQ(csr->row_ptr[1], 2);  // Row 0 has 2 entries
  EXPECT_EQ(csr->row_ptr[2], 3);  // Row 1 has 1 entry
  EXPECT_EQ(csr->row_ptr[3], 5);  // Row 2 has 2 entries

  // Check values using get
  matgen_value_t value;
  EXPECT_EQ(matgen_csr_get(csr, 0, 0, &value), MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, 1.0);

  EXPECT_EQ(matgen_csr_get(csr, 0, 2, &value), MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, 2.0);

  EXPECT_EQ(matgen_csr_get(csr, 1, 1, &value), MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, 3.0);

  EXPECT_EQ(matgen_csr_get(csr, 2, 0, &value), MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, 4.0);

  EXPECT_EQ(matgen_csr_get(csr, 2, 2, &value), MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, 5.0);
}

TEST_F(ConversionTest, COOtoCSR_UnsortedCOO) {
  coo = matgen_coo_create(3, 3, 3);
  ASSERT_NE(coo, nullptr);

  // Add in unsorted order
  matgen_coo_add_entry(coo, 2, 1, 3.0);
  matgen_coo_add_entry(coo, 0, 0, 1.0);
  matgen_coo_add_entry(coo, 1, 2, 2.0);

  csr = matgen_coo_to_csr(coo);
  ASSERT_NE(csr, nullptr);

  // Should still produce valid CSR
  EXPECT_TRUE(matgen_csr_validate(csr));

  // Check values are correct
  matgen_value_t value;
  EXPECT_EQ(matgen_csr_get(csr, 0, 0, &value), MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, 1.0);

  EXPECT_EQ(matgen_csr_get(csr, 1, 2, &value), MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, 2.0);

  EXPECT_EQ(matgen_csr_get(csr, 2, 1, &value), MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, 3.0);
}

TEST_F(ConversionTest, COOtoCSR_EmptyMatrix) {
  coo = matgen_coo_create(5, 5, 0);
  ASSERT_NE(coo, nullptr);

  csr = matgen_coo_to_csr(coo);
  ASSERT_NE(csr, nullptr);

  EXPECT_EQ(csr->nnz, 0);
  EXPECT_TRUE(matgen_csr_validate(csr));

  // All row_ptr should be 0
  for (matgen_index_t i = 0; i <= 5; i++) {
    EXPECT_EQ(csr->row_ptr[i], 0);
  }
}

TEST_F(ConversionTest, COOtoCSR_SingleEntry) {
  coo = matgen_coo_create(10, 10, 1);
  ASSERT_NE(coo, nullptr);

  matgen_coo_add_entry(coo, 5, 7, 42.0);

  csr = matgen_coo_to_csr(coo);
  ASSERT_NE(csr, nullptr);

  EXPECT_EQ(csr->nnz, 1);
  EXPECT_TRUE(matgen_csr_validate(csr));

  matgen_value_t value;
  EXPECT_EQ(matgen_csr_get(csr, 5, 7, &value), MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, 42.0);
}

TEST_F(ConversionTest, COOtoCSR_NullInput) {
  csr = matgen_coo_to_csr(nullptr);
  EXPECT_EQ(csr, nullptr);
}

TEST_F(ConversionTest, COOtoCSR_NonSquareMatrix) {
  coo = matgen_coo_create(2, 5, 3);
  ASSERT_NE(coo, nullptr);

  matgen_coo_add_entry(coo, 0, 1, 1.0);
  matgen_coo_add_entry(coo, 0, 4, 2.0);
  matgen_coo_add_entry(coo, 1, 2, 3.0);

  csr = matgen_coo_to_csr(coo);
  ASSERT_NE(csr, nullptr);

  EXPECT_EQ(csr->rows, 2);
  EXPECT_EQ(csr->cols, 5);
  EXPECT_TRUE(matgen_csr_validate(csr));
}

// =============================================================================
// CSR to COO Tests
// =============================================================================

TEST_F(ConversionTest, CSRtoCOO_ValidMatrix) {
  // Create CSR manually
  csr = matgen_csr_create(3, 3, 3);
  ASSERT_NE(csr, nullptr);

  // [1 0 0]
  // [0 2 0]
  // [0 0 3]
  csr->row_ptr[0] = 0;
  csr->row_ptr[1] = 1;
  csr->row_ptr[2] = 2;
  csr->row_ptr[3] = 3;

  csr->col_indices[0] = 0;
  csr->col_indices[1] = 1;
  csr->col_indices[2] = 2;

  csr->values[0] = 1.0;
  csr->values[1] = 2.0;
  csr->values[2] = 3.0;

  coo = matgen_csr_to_coo(csr);
  ASSERT_NE(coo, nullptr);

  EXPECT_EQ(coo->rows, 3);
  EXPECT_EQ(coo->cols, 3);
  EXPECT_EQ(coo->nnz, 3);
  EXPECT_TRUE(coo->is_sorted);
}

TEST_F(ConversionTest, CSRtoCOO_CheckValues) {
  csr = matgen_csr_create(3, 3, 3);
  ASSERT_NE(csr, nullptr);

  csr->row_ptr[0] = 0;
  csr->row_ptr[1] = 1;
  csr->row_ptr[2] = 2;
  csr->row_ptr[3] = 3;

  csr->col_indices[0] = 0;
  csr->col_indices[1] = 1;
  csr->col_indices[2] = 2;

  csr->values[0] = 1.0;
  csr->values[1] = 2.0;
  csr->values[2] = 3.0;

  coo = matgen_csr_to_coo(csr);
  ASSERT_NE(coo, nullptr);

  matgen_value_t value;
  EXPECT_EQ(matgen_coo_get(coo, 0, 0, &value), MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, 1.0);

  EXPECT_EQ(matgen_coo_get(coo, 1, 1, &value), MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, 2.0);

  EXPECT_EQ(matgen_coo_get(coo, 2, 2, &value), MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, 3.0);
}

TEST_F(ConversionTest, CSRtoCOO_EmptyMatrix) {
  csr = matgen_csr_create(5, 5, 0);
  ASSERT_NE(csr, nullptr);

  coo = matgen_csr_to_coo(csr);
  ASSERT_NE(coo, nullptr);

  EXPECT_EQ(coo->nnz, 0);
  EXPECT_TRUE(coo->is_sorted);
}

TEST_F(ConversionTest, CSRtoCOO_NullInput) {
  coo = matgen_csr_to_coo(nullptr);
  EXPECT_EQ(coo, nullptr);
}

// =============================================================================
// Round-trip Tests
// =============================================================================

TEST_F(ConversionTest, RoundTrip_COOtoCSRtoCOO) {
  coo = create_test_coo();
  ASSERT_NE(coo, nullptr);
  matgen_coo_sort(coo);

  // COO -> CSR
  csr = matgen_coo_to_csr(coo);
  ASSERT_NE(csr, nullptr);

  // CSR -> COO
  matgen_coo_matrix_t* coo2 = matgen_csr_to_coo(csr);
  ASSERT_NE(coo2, nullptr);

  // Should match original
  EXPECT_EQ(coo2->rows, coo->rows);
  EXPECT_EQ(coo2->cols, coo->cols);
  EXPECT_EQ(coo2->nnz, coo->nnz);

  // Check all values match
  for (matgen_size_t i = 0; i < coo->nnz; i++) {
    EXPECT_EQ(coo2->row_indices[i], coo->row_indices[i]);
    EXPECT_EQ(coo2->col_indices[i], coo->col_indices[i]);
    EXPECT_DOUBLE_EQ(coo2->values[i], coo->values[i]);
  }

  matgen_coo_destroy(coo2);
}

TEST_F(ConversionTest, RoundTrip_PreservesZeros) {
  coo = matgen_coo_create(10, 10, 0);
  ASSERT_NE(coo, nullptr);

  csr = matgen_coo_to_csr(coo);
  ASSERT_NE(csr, nullptr);

  matgen_coo_matrix_t* coo2 = matgen_csr_to_coo(csr);
  ASSERT_NE(coo2, nullptr);

  EXPECT_EQ(coo2->nnz, 0);

  matgen_coo_destroy(coo2);
}
