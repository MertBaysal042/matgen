#ifndef MATGEN_BACKENDS_MPI_INTERNAL_CONVERSION_MPI_H
#define MATGEN_BACKENDS_MPI_INTERNAL_CONVERSION_MPI_H

/**
 * @file conversion_mpi.h
 * @brief Internal header for MPI-distributed matrix format conversion
 *
 * This is an internal header used only by the library implementation.
 * Users should use the public API in <matgen/core/matrix/conversion.h> instead.
 *
 * MPI Conversion Strategy:
 *   - COO → CSR: Local conversion on each rank (assumes proper distribution)
 *   - CSR → COO: Local conversion on each rank
 *   - Both preserve row-wise distribution across ranks
 */

#include <mpi.h>

#include "matgen/core/matrix/coo.h"
#include "matgen/core/matrix/csr.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// MPI Conversion Operations
// =============================================================================

/**
 * @brief Convert distributed COO matrix to CSR format (MPI)
 *
 * Each rank converts its local COO portion to CSR independently.
 * Assumes COO is already sorted and properly distributed by rows.
 *
 * @param coo Source distributed COO matrix (local portion)
 * @return New local CSR matrix portion, or NULL on error
 */
matgen_csr_matrix_t* matgen_coo_to_csr_mpi(const matgen_coo_matrix_t* coo);

/**
 * @brief Convert distributed CSR matrix to COO format (MPI)
 *
 * Each rank converts its local CSR portion to COO independently.
 * Result is automatically sorted (CSR is sorted by definition).
 *
 * @param csr Source distributed CSR matrix (local portion)
 * @return New local COO matrix portion, or NULL on error
 */
matgen_coo_matrix_t* matgen_csr_to_coo_mpi(const matgen_csr_matrix_t* csr);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_BACKENDS_MPI_INTERNAL_CONVERSION_MPI_H
