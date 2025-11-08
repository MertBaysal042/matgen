#ifndef MATGEN_ALGORITHMS_SCALING_BILINEAR_OMP_H
#define MATGEN_ALGORITHMS_SCALING_BILINEAR_OMP_H

/**
 * @file bilinear_omp.h
 * @brief Bilinear interpolation for sparse matrix scaling (OpenMP parallel)
 */

#ifdef MATGEN_HAS_OPENMP

#include "matgen/core/csr_matrix.h"
#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Scale sparse matrix using bilinear interpolation (OpenMP parallel)
 *
 * Distributes each source entry's value to neighboring target cells
 * using bilinear weights. Values are accumulated (summed) at each target cell.
 *
 * This is a parallel implementation using OpenMP. Each thread processes a
 * subset of source rows using thread-local accumulators, which are then
 * merged at the end.
 *
 * @param source Source matrix (CSR format)
 * @param new_rows Target number of rows
 * @param new_cols Target number of columns
 * @param result Output: scaled matrix (CSR format)
 * @return MATGEN_SUCCESS on success, error code otherwise
 *
 * @note OpenMP parallel implementation
 * @note The number of threads can be controlled via OMP_NUM_THREADS environment
 *       variable or omp_set_num_threads()
 */
matgen_error_t matgen_scale_bilinear_omp(const matgen_csr_matrix_t* source,
                                         matgen_index_t new_rows,
                                         matgen_index_t new_cols,
                                         matgen_csr_matrix_t** result);

#ifdef __cplusplus
}
#endif

#endif

#endif  // MATGEN_ALGORITHMS_SCALING_BILINEAR_OMP_H
