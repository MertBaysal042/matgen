#ifndef MATGEN_GENERATORS_RANDOM_H
#define MATGEN_GENERATORS_RANDOM_H

/**
 * @file random.h
 * @brief Random sparse matrix generation
 *
 * Provides functions for generating random sparse matrices with various
 * distributions and patterns for testing and benchmarking.
 */

#include "matgen/core/coo_matrix.h"
#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Value distribution types
 */
typedef enum {
  MATGEN_DIST_UNIFORM,  // Uniform distribution in [min, max]
  MATGEN_DIST_NORMAL,   // Normal/Gaussian distribution
  MATGEN_DIST_CONSTANT  // All values equal to constant
} matgen_distribution_t;

/**
 * @brief Configuration for random matrix generation
 */
typedef struct {
  matgen_index_t rows;  // Number of rows
  matgen_index_t cols;  // Number of columns
  matgen_size_t nnz;    // Number of non-zeros to generate
  matgen_value_t
      density;  // Alternative: density (0.0-1.0), overrides nnz if > 0

  matgen_distribution_t distribution;  // Value distribution type

  // Uniform distribution parameters
  matgen_value_t min_value;  // Minimum value (uniform)
  matgen_value_t max_value;  // Maximum value (uniform)

  // Normal distribution parameters
  matgen_value_t mean;    // Mean (normal)
  matgen_value_t stddev;  // Standard deviation (normal)

  // Constant distribution
  matgen_value_t constant_value;  // Constant value

  u32 seed;  // Random seed (0 = use time-based seed)

  bool allow_duplicates;  // Allow duplicate (row,col) pairs?
  bool sorted;            // Sort output by (row, col)?
} matgen_random_config_t;

/**
 * @brief Initialize random config with default values
 *
 * Defaults:
 * - distribution: MATGEN_DIST_UNIFORM
 * - min_value: 0.0, max_value: 1.0
 * - mean: 0.0, stddev: 1.0
 * - constant_value: 1.0
 * - seed: 0 (time-based)
 * - allow_duplicates: false
 * - sorted: true
 *
 * @param config Config to initialize
 * @param rows Number of rows
 * @param cols Number of columns
 * @param nnz Number of non-zeros
 */
void matgen_random_config_init(matgen_random_config_t* config,
                               matgen_index_t rows, matgen_index_t cols,
                               matgen_size_t nnz);

/**
 * @brief Generate a random sparse COO matrix
 *
 * Generates a random sparse matrix according to the configuration.
 * If allow_duplicates is false and it's not possible to generate
 * the requested number of unique entries, returns NULL.
 *
 * @param config Configuration for generation
 * @return New COO matrix, or NULL on error
 */
matgen_coo_matrix_t* matgen_random_generate(
    const matgen_random_config_t* config);

/**
 * @brief Generate a random diagonal matrix
 *
 * Creates a diagonal matrix with random values on the diagonal.
 * Size will be min(rows, cols) non-zeros.
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @param distribution Value distribution
 * @param min_value Minimum value (uniform) or mean (normal)
 * @param max_value Maximum value (uniform) or stddev (normal)
 * @param seed Random seed (0 = time-based)
 * @return New COO matrix, or NULL on error
 */
matgen_coo_matrix_t* matgen_random_diagonal(matgen_index_t rows,
                                            matgen_index_t cols,
                                            matgen_distribution_t distribution,
                                            matgen_value_t min_value,
                                            matgen_value_t max_value, u32 seed);

/**
 * @brief Generate a random tridiagonal matrix
 *
 * Creates a tridiagonal matrix (main diagonal + upper + lower).
 *
 * @param size Matrix size (square matrix)
 * @param distribution Value distribution
 * @param min_value Minimum value (uniform) or mean (normal)
 * @param max_value Maximum value (uniform) or stddev (normal)
 * @param seed Random seed (0 = time-based)
 * @return New COO matrix, or NULL on error
 */
matgen_coo_matrix_t* matgen_random_tridiagonal(
    matgen_index_t size, matgen_distribution_t distribution,
    matgen_value_t min_value, matgen_value_t max_value, u32 seed);

/**
 * @brief Generate a random sparse matrix with specified sparsity pattern
 *
 * Similar to matgen_random_generate but with a simpler interface.
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @param nnz Number of non-zeros
 * @param seed Random seed (0 = time-based)
 * @return New COO matrix with random values in [0,1], or NULL on error
 */
matgen_coo_matrix_t* matgen_random_uniform(matgen_index_t rows,
                                           matgen_index_t cols,
                                           matgen_size_t nnz, u32 seed);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_GENERATORS_RANDOM_H
