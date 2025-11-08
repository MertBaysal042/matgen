#ifndef MATGEN_UTILS_TRIPLET_BUFFER_H
#define MATGEN_UTILS_TRIPLET_BUFFER_H

/**
 * @file triplet_buffer.h
 * @brief Thread-safe dynamic buffer for COO triplets (row, col, value)
 *
 * Provides an efficient way to collect matrix entries during parallel
 * construction, particularly useful for OpenMP implementations.
 */

#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Types
// =============================================================================

/**
 * @brief Dynamic buffer for storing COO matrix triplets
 *
 * Thread-local buffer that efficiently collects (row, col, value) triplets
 * with automatic resizing. Optimized for append operations.
 */
typedef struct {
  matgen_index_t* rows; /**< Row indices */
  matgen_index_t* cols; /**< Column indices */
  matgen_value_t* vals; /**< Values */
  size_t capacity;      /**< Current capacity */
  size_t size;          /**< Number of entries */
} matgen_triplet_buffer_t;

// =============================================================================
// Core API
// =============================================================================

/**
 * @brief Create a new triplet buffer with specified initial capacity
 *
 * @param initial_capacity Initial capacity (number of triplets)
 * @return Pointer to new buffer, or NULL on allocation failure
 *
 * @note The buffer automatically resizes when full
 */
matgen_triplet_buffer_t* matgen_triplet_buffer_create(size_t initial_capacity);

/**
 * @brief Destroy a triplet buffer and free its memory
 *
 * @param buffer Buffer to destroy (NULL safe)
 */
void matgen_triplet_buffer_destroy(matgen_triplet_buffer_t* buffer);

/**
 * @brief Add a triplet to the buffer
 *
 * Automatically resizes the buffer if necessary (doubles capacity).
 *
 * @param buffer Buffer to add to
 * @param row Row index
 * @param col Column index
 * @param val Value
 * @return MATGEN_SUCCESS on success, error code otherwise
 */
matgen_error_t matgen_triplet_buffer_add(matgen_triplet_buffer_t* buffer,
                                         matgen_index_t row, matgen_index_t col,
                                         matgen_value_t val);

/**
 * @brief Clear all entries from the buffer without deallocating
 *
 * Resets size to 0 but keeps allocated capacity.
 *
 * @param buffer Buffer to clear
 */
void matgen_triplet_buffer_clear(matgen_triplet_buffer_t* buffer);

/**
 * @brief Get the current number of triplets in the buffer
 *
 * @param buffer Buffer to query
 * @return Number of triplets (0 if buffer is NULL)
 */
size_t matgen_triplet_buffer_size(const matgen_triplet_buffer_t* buffer);

/**
 * @brief Get the current capacity of the buffer
 *
 * @param buffer Buffer to query
 * @return Capacity (0 if buffer is NULL)
 */
size_t matgen_triplet_buffer_capacity(const matgen_triplet_buffer_t* buffer);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_UTILS_TRIPLET_BUFFER_H
