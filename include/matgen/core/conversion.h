#ifndef MATGEN_CORE_CONVERSION_H
#define MATGEN_CORE_CONVERSION_H

/**
 * @file conversion.h
 * @brief Format conversion functions for sparse matrices
 *
 * Converts between COO (Coordinate) and CSR (Compressed Sparse Row) formats.
 */

#include "matgen/core/coo_matrix.h"
#include "matgen/core/csr_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Convert COO matrix to CSR format
 *
 * The COO matrix will be sorted during conversion if not already sorted.
 * The original COO matrix is not modified (sorting is done on a copy if
 * needed).
 *
 * @param coo Input COO matrix (const, not modified)
 * @return New CSR matrix, or NULL on error
 */
matgen_csr_matrix_t* matgen_coo_to_csr(const matgen_coo_matrix_t* coo);

/**
 * @brief Convert CSR matrix to COO format
 *
 * @param csr Input CSR matrix
 * @return New COO matrix, or NULL on error
 */
matgen_coo_matrix_t* matgen_csr_to_coo(const matgen_csr_matrix_t* csr);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_CORE_CONVERSION_H
