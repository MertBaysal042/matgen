#include "matgen/utils/triplet_buffer.h"

#include <stdlib.h>
#include <string.h>

#include "matgen/utils/log.h"

// Default initial capacity if 0 is specified
#define MATGEN_TRIPLET_BUFFER_DEFAULT_CAPACITY 1024

matgen_triplet_buffer_t* matgen_triplet_buffer_create(size_t initial_capacity) {
  // Use default if 0 specified
  if (initial_capacity == 0) {
    initial_capacity = MATGEN_TRIPLET_BUFFER_DEFAULT_CAPACITY;
  }

  matgen_triplet_buffer_t* buffer =
      (matgen_triplet_buffer_t*)malloc(sizeof(matgen_triplet_buffer_t));
  if (!buffer) {
    MATGEN_LOG_ERROR("Failed to allocate triplet buffer structure");
    return NULL;
  }

  buffer->rows =
      (matgen_index_t*)malloc(initial_capacity * sizeof(matgen_index_t));
  buffer->cols =
      (matgen_index_t*)malloc(initial_capacity * sizeof(matgen_index_t));
  buffer->vals =
      (matgen_value_t*)malloc(initial_capacity * sizeof(matgen_value_t));

  if (!buffer->rows || !buffer->cols || !buffer->vals) {
    MATGEN_LOG_ERROR("Failed to allocate triplet buffer arrays (capacity: %zu)",
                     initial_capacity);
    free(buffer->rows);
    free(buffer->cols);
    free(buffer->vals);
    free(buffer);
    return NULL;
  }

  buffer->capacity = initial_capacity;
  buffer->size = 0;

  return buffer;
}

void matgen_triplet_buffer_destroy(matgen_triplet_buffer_t* buffer) {
  if (buffer) {
    free(buffer->rows);
    free(buffer->cols);
    free(buffer->vals);
    free(buffer);
  }
}

matgen_error_t matgen_triplet_buffer_add(matgen_triplet_buffer_t* buffer,
                                         matgen_index_t row, matgen_index_t col,
                                         matgen_value_t val) {
  if (!buffer) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  // Resize if needed (double capacity)
  if (buffer->size >= buffer->capacity) {
    size_t new_capacity = buffer->capacity * 2;

    matgen_index_t* new_rows = (matgen_index_t*)realloc(
        buffer->rows, new_capacity * sizeof(matgen_index_t));
    matgen_index_t* new_cols = (matgen_index_t*)realloc(
        buffer->cols, new_capacity * sizeof(matgen_index_t));
    matgen_value_t* new_vals = (matgen_value_t*)realloc(
        buffer->vals, new_capacity * sizeof(matgen_value_t));

    if (!new_rows || !new_cols || !new_vals) {
      // Cleanup on partial failure
      free(new_rows);
      free(new_cols);
      free(new_vals);
      MATGEN_LOG_ERROR(
          "Failed to resize triplet buffer from %zu to %zu entries",
          buffer->capacity, new_capacity);
      return MATGEN_ERROR_OUT_OF_MEMORY;
    }

    buffer->rows = new_rows;
    buffer->cols = new_cols;
    buffer->vals = new_vals;
    buffer->capacity = new_capacity;
  }

  // Add the triplet
  buffer->rows[buffer->size] = row;
  buffer->cols[buffer->size] = col;
  buffer->vals[buffer->size] = val;
  buffer->size++;

  return MATGEN_SUCCESS;
}

void matgen_triplet_buffer_clear(matgen_triplet_buffer_t* buffer) {
  if (buffer) {
    buffer->size = 0;
  }
}

size_t matgen_triplet_buffer_size(const matgen_triplet_buffer_t* buffer) {
  return buffer ? buffer->size : 0;
}

size_t matgen_triplet_buffer_capacity(const matgen_triplet_buffer_t* buffer) {
  return buffer ? buffer->capacity : 0;
}
