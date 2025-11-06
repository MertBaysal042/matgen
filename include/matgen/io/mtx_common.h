#ifndef MATGEN_IO_MTX_COMMON_H
#define MATGEN_IO_MTX_COMMON_H

/**
 * @file mtx_common.h
 * @brief Common definitions for Matrix Market format I/O
 *
 * Matrix Market is a standard ASCII format for sparse matrices.
 * @see https://math.nist.gov/MatrixMarket/formats.html
 */

#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Matrix Market object types
 */
typedef enum { MATGEN_MM_MATRIX, MATGEN_MM_VECTOR } matgen_mm_object_t;

/**
 * @brief Matrix Market format types
 */
typedef enum {
  MATGEN_MM_COORDINATE,  // Sparse coordinate format
  MATGEN_MM_ARRAY        // Dense array format
} matgen_mm_format_t;

/**
 * @brief Matrix Market value types
 */
typedef enum {
  MATGEN_MM_REAL,     // Real valued
  MATGEN_MM_INTEGER,  // Integer valued
  MATGEN_MM_PATTERN,  // Pattern (no values, all ones)
  MATGEN_MM_COMPLEX   // Complex valued (not yet supported)
} matgen_mm_value_type_t;

/**
 * @brief Matrix Market symmetry types
 */
typedef enum {
  MATGEN_MM_GENERAL,         // No symmetry
  MATGEN_MM_SYMMETRIC,       // Symmetric matrix (A = A^T)
  MATGEN_MM_SKEW_SYMMETRIC,  // Skew-symmetric (A = -A^T)
  MATGEN_MM_HERMITIAN        // Hermitian (not yet supported)
} matgen_mm_symmetry_t;

/**
 * @brief Matrix Market file information
 */
typedef struct {
  matgen_mm_object_t object;          // Object type
  matgen_mm_format_t format;          // Format type
  matgen_mm_value_type_t value_type;  // Value type
  matgen_mm_symmetry_t symmetry;      // Symmetry type

  matgen_index_t rows;  // Number of rows
  matgen_index_t cols;  // Number of columns
  matgen_size_t nnz;    // Number of non-zeros (as stored in file)
} matgen_mm_info_t;

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_IO_MTX_COMMON_H
