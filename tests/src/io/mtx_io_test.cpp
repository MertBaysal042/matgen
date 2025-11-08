#include <gtest/gtest.h>
#include <matgen/core/conversion.h>
#include <matgen/core/coo_matrix.h>
#include <matgen/core/csr_matrix.h>
#include <matgen/core/types.h>
#include <matgen/io/mtx_common.h>
#include <matgen/io/mtx_reader.h>
#include <matgen/io/mtx_writer.h>

#include <fstream>
#include <string>

// Test fixture for MTX I/O
class MTXIOTest : public ::testing::Test {
 protected:
  matgen_coo_matrix_t* coo;  // NOLINT
  matgen_csr_matrix_t* csr;  // NOLINT
  std::string temp_file;     // NOLINT

  void SetUp() override {
    coo = nullptr;
    csr = nullptr;
    temp_file = "test_matrix.mtx";
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

    // Clean up temp file
    std::remove(temp_file.c_str());
  }

  // Helper: Create a simple test matrix
  // [1 0 2]
  // [0 3 0]
  // [4 0 5]
  void create_test_mtx_file() {
    std::ofstream f(temp_file);
    f << "%%MatrixMarket matrix coordinate real general\n";
    f << "% Test matrix\n";
    f << "3 3 5\n";
    f << "1 1 1.0\n";
    f << "1 3 2.0\n";
    f << "2 2 3.0\n";
    f << "3 1 4.0\n";
    f << "3 3 5.0\n";
    f.close();
  }

  // Helper: Create symmetric matrix file
  void create_symmetric_mtx_file() {
    std::ofstream f(temp_file);
    f << "%%MatrixMarket matrix coordinate real symmetric\n";
    f << "3 3 4\n";
    f << "1 1 1.0\n";
    f << "2 2 2.0\n";
    f << "2 3 3.0\n";  // Will expand to (2,3) and (3,2)
    f << "3 3 4.0\n";
    f.close();
  }

  // Helper: Create pattern matrix file
  void create_pattern_mtx_file() {
    std::ofstream f(temp_file);
    f << "%%MatrixMarket matrix coordinate pattern general\n";
    f << "3 3 3\n";
    f << "1 1\n";
    f << "2 2\n";
    f << "3 3\n";
    f.close();
  }

  // Helper: Create integer matrix file
  void create_integer_mtx_file() {
    std::ofstream f(temp_file);
    f << "%%MatrixMarket matrix coordinate integer general\n";
    f << "2 2 2\n";
    f << "1 1 10\n";
    f << "2 2 20\n";
    f.close();
  }
};

// =============================================================================
// Header Reading Tests
// =============================================================================

TEST_F(MTXIOTest, ReadHeader_ValidFile) {
  create_test_mtx_file();

  matgen_mm_info_t info;
  matgen_error_t err = matgen_mtx_read_header(temp_file.c_str(), &info);

  EXPECT_EQ(err, MATGEN_SUCCESS);
  EXPECT_EQ(info.object, MATGEN_MM_MATRIX);
  EXPECT_EQ(info.format, MATGEN_MM_COORDINATE);
  EXPECT_EQ(info.value_type, MATGEN_MM_REAL);
  EXPECT_EQ(info.symmetry, MATGEN_MM_GENERAL);
  EXPECT_EQ(info.rows, 3);
  EXPECT_EQ(info.cols, 3);
  EXPECT_EQ(info.nnz, 5);
}

TEST_F(MTXIOTest, ReadHeader_SymmetricFile) {
  create_symmetric_mtx_file();

  matgen_mm_info_t info;
  matgen_error_t err = matgen_mtx_read_header(temp_file.c_str(), &info);

  EXPECT_EQ(err, MATGEN_SUCCESS);
  EXPECT_EQ(info.symmetry, MATGEN_MM_SYMMETRIC);
  EXPECT_EQ(info.nnz, 4);  // As stored, not expanded
}

TEST_F(MTXIOTest, ReadHeader_PatternFile) {
  create_pattern_mtx_file();

  matgen_mm_info_t info;
  matgen_error_t err = matgen_mtx_read_header(temp_file.c_str(), &info);

  EXPECT_EQ(err, MATGEN_SUCCESS);
  EXPECT_EQ(info.value_type, MATGEN_MM_PATTERN);
}

TEST_F(MTXIOTest, ReadHeader_NonExistentFile) {
  matgen_mm_info_t info;
  matgen_error_t err = matgen_mtx_read_header("nonexistent.mtx", &info);

  EXPECT_EQ(err, MATGEN_ERROR_IO);
}

TEST_F(MTXIOTest, ReadHeader_NullArguments) {
  matgen_mm_info_t info;

  EXPECT_EQ(matgen_mtx_read_header(nullptr, &info),
            MATGEN_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(matgen_mtx_read_header(temp_file.c_str(), nullptr),
            MATGEN_ERROR_INVALID_ARGUMENT);
}

TEST_F(MTXIOTest, ReadHeader_InvalidFormat) {
  std::ofstream f(temp_file);
  f << "Not a valid matrix market file\n";
  f.close();

  matgen_mm_info_t info;
  matgen_error_t err = matgen_mtx_read_header(temp_file.c_str(), &info);

  EXPECT_NE(err, MATGEN_SUCCESS);
}

// =============================================================================
// Reading Tests
// =============================================================================

TEST_F(MTXIOTest, Read_GeneralMatrix) {
  create_test_mtx_file();

  matgen_mm_info_t info;
  coo = matgen_mtx_read(temp_file.c_str(), &info);

  ASSERT_NE(coo, nullptr);
  EXPECT_EQ(coo->rows, 3);
  EXPECT_EQ(coo->cols, 3);
  EXPECT_EQ(coo->nnz, 5);
  EXPECT_TRUE(coo->is_sorted);

  // Check values
  matgen_value_t value;
  EXPECT_EQ(matgen_coo_get(coo, 0, 0, &value), MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, 1.0);

  EXPECT_EQ(matgen_coo_get(coo, 0, 2, &value), MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, 2.0);

  EXPECT_EQ(matgen_coo_get(coo, 1, 1, &value), MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, 3.0);

  EXPECT_EQ(matgen_coo_get(coo, 2, 0, &value), MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, 4.0);

  EXPECT_EQ(matgen_coo_get(coo, 2, 2, &value), MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, 5.0);
}

TEST_F(MTXIOTest, Read_SymmetricMatrix) {
  create_symmetric_mtx_file();

  coo = matgen_mtx_read(temp_file.c_str(), nullptr);

  ASSERT_NE(coo, nullptr);
  EXPECT_EQ(coo->rows, 3);
  EXPECT_EQ(coo->cols, 3);

  // Should be expanded: 4 stored -> 5 actual (one off-diagonal expanded)
  EXPECT_EQ(coo->nnz, 5);

  // Check symmetric entries exist
  matgen_value_t value;
  EXPECT_EQ(matgen_coo_get(coo, 1, 2, &value), MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, 3.0);

  EXPECT_EQ(matgen_coo_get(coo, 2, 1, &value), MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, 3.0);  // Transpose
}

TEST_F(MTXIOTest, Read_PatternMatrix) {
  create_pattern_mtx_file();

  coo = matgen_mtx_read(temp_file.c_str(), nullptr);

  ASSERT_NE(coo, nullptr);
  EXPECT_EQ(coo->nnz, 3);

  // All values should be 1.0
  for (matgen_size_t i = 0; i < coo->nnz; i++) {
    EXPECT_DOUBLE_EQ(coo->values[i], 1.0);
  }
}

TEST_F(MTXIOTest, Read_IntegerMatrix) {
  create_integer_mtx_file();

  coo = matgen_mtx_read(temp_file.c_str(), nullptr);

  ASSERT_NE(coo, nullptr);
  EXPECT_EQ(coo->nnz, 2);

  matgen_value_t value;
  EXPECT_EQ(matgen_coo_get(coo, 0, 0, &value), MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, 10.0);

  EXPECT_EQ(matgen_coo_get(coo, 1, 1, &value), MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, 20.0);
}

TEST_F(MTXIOTest, Read_NullFilename) {
  coo = matgen_mtx_read(nullptr, nullptr);
  EXPECT_EQ(coo, nullptr);
}

TEST_F(MTXIOTest, Read_NonExistentFile) {
  coo = matgen_mtx_read("nonexistent.mtx", nullptr);
  EXPECT_EQ(coo, nullptr);
}

TEST_F(MTXIOTest, Read_EmptyMatrix) {
  std::ofstream f(temp_file);
  f << "%%MatrixMarket matrix coordinate real general\n";
  f << "5 5 0\n";
  f.close();

  coo = matgen_mtx_read(temp_file.c_str(), nullptr);

  ASSERT_NE(coo, nullptr);
  EXPECT_EQ(coo->rows, 5);
  EXPECT_EQ(coo->cols, 5);
  EXPECT_EQ(coo->nnz, 0);
}

TEST_F(MTXIOTest, Read_NonSquareMatrix) {
  std::ofstream f(temp_file);
  f << "%%MatrixMarket matrix coordinate real general\n";
  f << "2 5 3\n";
  f << "1 2 1.5\n";
  f << "2 4 2.5\n";
  f << "1 5 3.5\n";
  f.close();

  coo = matgen_mtx_read(temp_file.c_str(), nullptr);

  ASSERT_NE(coo, nullptr);
  EXPECT_EQ(coo->rows, 2);
  EXPECT_EQ(coo->cols, 5);
  EXPECT_EQ(coo->nnz, 3);
}

// =============================================================================
// Writing Tests
// =============================================================================

TEST_F(MTXIOTest, WriteCOO_ValidMatrix) {
  coo = matgen_coo_create(3, 3, 3);
  ASSERT_NE(coo, nullptr);

  matgen_coo_add_entry(coo, 0, 0, 1.5);
  matgen_coo_add_entry(coo, 1, 1, 2.5);
  matgen_coo_add_entry(coo, 2, 2, 3.5);

  matgen_error_t err = matgen_mtx_write_coo(temp_file.c_str(), coo);
  EXPECT_EQ(err, MATGEN_SUCCESS);

  // Verify file exists and is readable
  std::ifstream f(temp_file);
  EXPECT_TRUE(f.good());

  // Read it back
  matgen_coo_matrix_t* coo2 = matgen_mtx_read(temp_file.c_str(), nullptr);
  ASSERT_NE(coo2, nullptr);

  EXPECT_EQ(coo2->rows, 3);
  EXPECT_EQ(coo2->cols, 3);
  EXPECT_EQ(coo2->nnz, 3);

  matgen_coo_destroy(coo2);
}

TEST_F(MTXIOTest, WriteCOO_EmptyMatrix) {
  coo = matgen_coo_create(5, 5, 0);
  ASSERT_NE(coo, nullptr);

  matgen_error_t err = matgen_mtx_write_coo(temp_file.c_str(), coo);
  EXPECT_EQ(err, MATGEN_SUCCESS);

  // Read it back
  matgen_coo_matrix_t* coo2 = matgen_mtx_read(temp_file.c_str(), nullptr);
  ASSERT_NE(coo2, nullptr);
  EXPECT_EQ(coo2->nnz, 0);

  matgen_coo_destroy(coo2);
}

TEST_F(MTXIOTest, WriteCOO_NullArguments) {
  coo = matgen_coo_create(3, 3, 1);

  EXPECT_EQ(matgen_mtx_write_coo(nullptr, coo), MATGEN_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(matgen_mtx_write_coo(temp_file.c_str(), nullptr),
            MATGEN_ERROR_INVALID_ARGUMENT);
}

TEST_F(MTXIOTest, WriteCOO_NonSquareMatrix) {
  coo = matgen_coo_create(2, 5, 2);
  ASSERT_NE(coo, nullptr);

  matgen_coo_add_entry(coo, 0, 1, 1.0);
  matgen_coo_add_entry(coo, 1, 4, 2.0);

  matgen_error_t err = matgen_mtx_write_coo(temp_file.c_str(), coo);
  EXPECT_EQ(err, MATGEN_SUCCESS);

  // Read it back
  matgen_coo_matrix_t* coo2 = matgen_mtx_read(temp_file.c_str(), nullptr);
  ASSERT_NE(coo2, nullptr);
  EXPECT_EQ(coo2->rows, 2);
  EXPECT_EQ(coo2->cols, 5);

  matgen_coo_destroy(coo2);
}

TEST_F(MTXIOTest, WriteCSR_ValidMatrix) {
  csr = matgen_csr_create(3, 3, 3);
  ASSERT_NE(csr, nullptr);

  // Diagonal matrix
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

  matgen_error_t err = matgen_mtx_write_csr(temp_file.c_str(), csr);
  EXPECT_EQ(err, MATGEN_SUCCESS);

  // Read it back
  matgen_coo_matrix_t* coo2 = matgen_mtx_read(temp_file.c_str(), nullptr);
  ASSERT_NE(coo2, nullptr);

  EXPECT_EQ(coo2->rows, 3);
  EXPECT_EQ(coo2->cols, 3);
  EXPECT_EQ(coo2->nnz, 3);

  matgen_coo_destroy(coo2);
}

TEST_F(MTXIOTest, WriteCSR_NullArguments) {
  csr = matgen_csr_create(3, 3, 1);

  EXPECT_EQ(matgen_mtx_write_csr(nullptr, csr), MATGEN_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(matgen_mtx_write_csr(temp_file.c_str(), nullptr),
            MATGEN_ERROR_INVALID_ARGUMENT);
}

// =============================================================================
// Round-trip Tests
// =============================================================================

TEST_F(MTXIOTest, RoundTrip_COO) {
  coo = matgen_coo_create(10, 10, 20);
  ASSERT_NE(coo, nullptr);

  // Add some entries
  for (matgen_index_t i = 0; i < 10; i++) {
    matgen_coo_add_entry(coo, i, i, (matgen_value_t)i);
    if (i < 9) {
      matgen_coo_add_entry(coo, i, i + 1, (matgen_value_t)(i + 10));
    }
  }

  matgen_coo_sort(coo);

  // Write
  EXPECT_EQ(matgen_mtx_write_coo(temp_file.c_str(), coo), MATGEN_SUCCESS);

  // Read back
  matgen_coo_matrix_t* coo2 = matgen_mtx_read(temp_file.c_str(), nullptr);
  ASSERT_NE(coo2, nullptr);

  // Compare
  EXPECT_EQ(coo2->rows, coo->rows);
  EXPECT_EQ(coo2->cols, coo->cols);
  EXPECT_EQ(coo2->nnz, coo->nnz);

  for (matgen_size_t i = 0; i < coo->nnz; i++) {
    EXPECT_EQ(coo2->row_indices[i], coo->row_indices[i]);
    EXPECT_EQ(coo2->col_indices[i], coo->col_indices[i]);
    EXPECT_DOUBLE_EQ(coo2->values[i], coo->values[i]);
  }

  matgen_coo_destroy(coo2);
}

TEST_F(MTXIOTest, RoundTrip_CSR) {
  coo = matgen_coo_create(5, 5, 10);
  ASSERT_NE(coo, nullptr);

  for (matgen_index_t i = 0; i < 5; i++) {
    matgen_coo_add_entry(coo, i, i, (matgen_value_t)i);
    if (i < 4) {
      matgen_coo_add_entry(coo, i, i + 1, (matgen_value_t)(i + 5));
    }
  }

  csr = matgen_coo_to_csr(coo);
  ASSERT_NE(csr, nullptr);

  // Write CSR
  EXPECT_EQ(matgen_mtx_write_csr(temp_file.c_str(), csr), MATGEN_SUCCESS);

  // Read back as COO
  matgen_coo_matrix_t* coo2 = matgen_mtx_read(temp_file.c_str(), nullptr);
  ASSERT_NE(coo2, nullptr);

  // Should match original COO
  EXPECT_EQ(coo2->rows, coo->rows);
  EXPECT_EQ(coo2->cols, coo->cols);
  EXPECT_EQ(coo2->nnz, coo->nnz);

  matgen_coo_destroy(coo2);
}

TEST_F(MTXIOTest, RoundTrip_LargeValues) {
  coo = matgen_coo_create(3, 3, 3);
  ASSERT_NE(coo, nullptr);

  matgen_coo_add_entry(coo, 0, 0, 1.23456789012345e100);
  matgen_coo_add_entry(coo, 1, 1, -9.87654321098765e-100);
  matgen_coo_add_entry(coo, 2, 2, 3.14159265358979);

  EXPECT_EQ(matgen_mtx_write_coo(temp_file.c_str(), coo), MATGEN_SUCCESS);

  matgen_coo_matrix_t* coo2 = matgen_mtx_read(temp_file.c_str(), nullptr);
  ASSERT_NE(coo2, nullptr);

  // Check precision is preserved reasonably
  for (matgen_size_t i = 0; i < 3; i++) {
    EXPECT_NEAR(coo2->values[i], coo->values[i],
                1e-10 * std::abs(coo->values[i]));
  }

  matgen_coo_destroy(coo2);
}
