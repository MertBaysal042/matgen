#include "backends/omp/internal/bilinear_omp.h"

#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#include "matgen/core/matrix/conversion.h"
#include "matgen/core/matrix/coo.h"
#include "matgen/core/types.h"
#include "matgen/utils/log.h"
#include "matgen/utils/triplet_buffer.h"

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
matgen_error_t matgen_scale_bilinear_omp(const matgen_csr_matrix_t* source,
                                         matgen_index_t new_rows,
                                         matgen_index_t new_cols,
                                         matgen_csr_matrix_t** result) {
  if (!source || !result) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (new_rows == 0 || new_cols == 0) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  *result = NULL;

  matgen_value_t row_scale =
      (matgen_value_t)new_rows / (matgen_value_t)source->rows;
  matgen_value_t col_scale =
      (matgen_value_t)new_cols / (matgen_value_t)source->cols;

  MATGEN_LOG_DEBUG(
      "Bilinear scaling (OMP): %zu×%zu -> %zu×%zu (scale: %.3fx%.3f)",
      source->rows, source->cols, new_rows, new_cols, row_scale, col_scale);

  int num_threads = omp_get_max_threads();
  MATGEN_LOG_DEBUG("Using %d OpenMP threads", num_threads);

  // Estimate output NNZ per thread
  matgen_value_t avg_contributions_per_source =
      (matgen_value_t)(row_scale + 1.0) * (matgen_value_t)(col_scale + 1.0);
  size_t estimated_nnz_per_thread =
      (size_t)((matgen_value_t)source->nnz * avg_contributions_per_source *
               1.2 / num_threads);

  MATGEN_LOG_DEBUG("Estimated output NNZ per thread: %zu",
                   estimated_nnz_per_thread);

  // Create thread-local triplet buffers
  matgen_triplet_buffer_t** thread_buffers = (matgen_triplet_buffer_t**)calloc(
      num_threads, sizeof(matgen_triplet_buffer_t*));
  if (!thread_buffers) {
    MATGEN_LOG_ERROR("Failed to allocate thread buffer array");
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  matgen_error_t err = MATGEN_SUCCESS;

  // Process rows in parallel with thread-local buffers
#pragma omp parallel
  {
    int tid = omp_get_thread_num();

    // Create thread-local buffer
    thread_buffers[tid] =
        matgen_triplet_buffer_create(estimated_nnz_per_thread);
    if (!thread_buffers[tid]) {
#pragma omp atomic write
      err = MATGEN_ERROR_OUT_OF_MEMORY;
    }

#pragma omp barrier

    if (err == MATGEN_SUCCESS) {
      int src_row;

#pragma omp for schedule(dynamic, 16)
      for (src_row = 0; src_row < source->rows; src_row++) {
        if (err != MATGEN_SUCCESS) {
          continue;
        }

        size_t row_start = source->row_ptr[src_row];
        size_t row_end = source->row_ptr[src_row + 1];

        for (size_t idx = row_start; idx < row_end; idx++) {
          matgen_index_t src_col = source->col_indices[idx];
          matgen_value_t src_val = source->values[idx];

          if (src_val == 0.0) {
            continue;
          }

          // For bilinear interpolation, calculate destination range
          matgen_value_t dst_row_start_f = (matgen_value_t)fmax(
              0.0, ((matgen_value_t)src_row - 1.0) * row_scale);
          matgen_value_t dst_row_end_f =
              (matgen_value_t)((matgen_value_t)src_row + 1.0) * row_scale;
          matgen_value_t dst_col_start_f = (matgen_value_t)fmax(
              0.0, ((matgen_value_t)src_col - 1.0) * col_scale);
          matgen_value_t dst_col_end_f =
              (matgen_value_t)((matgen_value_t)src_col + 1.0) * col_scale;

          matgen_index_t dst_row_start = (matgen_index_t)ceil(dst_row_start_f);
          matgen_index_t dst_row_end = (matgen_index_t)ceil(dst_row_end_f);
          matgen_index_t dst_col_start = (matgen_index_t)ceil(dst_col_start_f);
          matgen_index_t dst_col_end = (matgen_index_t)ceil(dst_col_end_f);

          dst_row_start = MATGEN_CLAMP(dst_row_start, 0, new_rows);
          dst_row_end = MATGEN_CLAMP(dst_row_end, 0, new_rows);
          dst_col_start = MATGEN_CLAMP(dst_col_start, 0, new_cols);
          dst_col_end = MATGEN_CLAMP(dst_col_end, 0, new_cols);

          // For each destination cell, calculate bilinear weight
          for (matgen_index_t dst_row = dst_row_start; dst_row < dst_row_end;
               dst_row++) {
            for (matgen_index_t dst_col = dst_col_start; dst_col < dst_col_end;
                 dst_col++) {
              // Map destination back to source coordinates (fractional)
              matgen_value_t src_y = (matgen_value_t)dst_row / row_scale;
              matgen_value_t src_x = (matgen_value_t)dst_col / col_scale;

              // Find the 4 bilinear neighbors
              matgen_index_t y0 = (matgen_index_t)floor(src_y);
              matgen_index_t y1 = (matgen_index_t)ceil(src_y);
              matgen_index_t x0 = (matgen_index_t)floor(src_x);
              matgen_index_t x1 = (matgen_index_t)ceil(src_x);

              // Clamp neighbors to valid source bounds
              y0 = MATGEN_CLAMP(y0, 0, source->rows - 1);
              y1 = MATGEN_CLAMP(y1, 0, source->rows - 1);
              x0 = MATGEN_CLAMP(x0, 0, source->cols - 1);
              x1 = MATGEN_CLAMP(x1, 0, source->cols - 1);

              // Calculate fractional parts for bilinear interpolation
              matgen_value_t dy = src_y - (matgen_value_t)y0;
              matgen_value_t dx = src_x - (matgen_value_t)x0;

              // Clamp fractional parts to [0, 1]
              dy = MATGEN_CLAMP(dy, 0.0, 1.0);
              dx = MATGEN_CLAMP(dx, 0.0, 1.0);

              // Determine which of the 4 neighbors we are and calculate
              // bilinear weight
              matgen_value_t weight = (matgen_value_t)0.0;

              if (src_row == y0 && src_col == x0) {
                // Bottom-left neighbor
                weight =
                    (matgen_value_t)(1.0 - dy) * (matgen_value_t)(1.0 - dx);
              } else if (src_row == y0 && src_col == x1) {
                // Bottom-right neighbor
                weight = (matgen_value_t)(1.0 - dy) * dx;
              } else if (src_row == y1 && src_col == x0) {
                // Top-left neighbor
                weight = dy * (matgen_value_t)(1.0 - dx);
              } else if (src_row == y1 && src_col == x1) {
                // Top-right neighbor
                weight = dy * dx;
              }

              // Add weighted contribution if non-zero
              if (weight > 1e-12) {
                matgen_value_t contribution = src_val * weight;
                matgen_error_t local_err = matgen_triplet_buffer_add(
                    thread_buffers[tid], dst_row, dst_col, contribution);
                if (local_err != MATGEN_SUCCESS) {
#pragma omp atomic write
                  err = local_err;
                  break;
                }
              }
            }
            if (err != MATGEN_SUCCESS) {
              break;
            }
          }
        }
      }
    }
  }

  if (err != MATGEN_SUCCESS) {
    MATGEN_LOG_ERROR("Error during parallel processing");
    for (int i = 0; i < num_threads; i++) {
      if (thread_buffers[i]) {
        matgen_triplet_buffer_destroy(thread_buffers[i]);
      }
    }
    free((void*)thread_buffers);
    return err;
  }

  // Merge all thread-local buffers into a single COO matrix
  size_t total_triplets = 0;
  for (int i = 0; i < num_threads; i++) {
    if (thread_buffers[i]) {
      total_triplets += matgen_triplet_buffer_size(thread_buffers[i]);
    }
  }

  MATGEN_LOG_DEBUG("Generated %zu total triplets from %d threads",
                   total_triplets, num_threads);

  // Create COO matrix
  matgen_coo_matrix_t* coo =
      matgen_coo_create(new_rows, new_cols, total_triplets);
  if (!coo) {
    MATGEN_LOG_ERROR("Failed to create COO matrix");
    for (int i = 0; i < num_threads; i++) {
      if (thread_buffers[i]) {
        matgen_triplet_buffer_destroy(thread_buffers[i]);
      }
    }
    free((void*)thread_buffers);
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  // Copy triplets from all thread buffers to COO
  size_t offset = 0;
  for (int i = 0; i < num_threads; i++) {
    if (thread_buffers[i]) {
      size_t thread_size = matgen_triplet_buffer_size(thread_buffers[i]);
      if (thread_size > 0) {
        memcpy(&coo->row_indices[offset], thread_buffers[i]->rows,
               thread_size * sizeof(matgen_index_t));
        memcpy(&coo->col_indices[offset], thread_buffers[i]->cols,
               thread_size * sizeof(matgen_index_t));
        memcpy(&coo->values[offset], thread_buffers[i]->vals,
               thread_size * sizeof(matgen_value_t));
        offset += thread_size;
      }
      matgen_triplet_buffer_destroy(thread_buffers[i]);
    }
  }
  free((void*)thread_buffers);

  coo->nnz = total_triplets;
  coo->is_sorted = false;

  // Sort and sum duplicates using OpenMP policy
  MATGEN_LOG_DEBUG("Sorting and summing duplicates...");
  err = matgen_coo_sort_with_policy(coo, MATGEN_EXEC_PAR);
  if (err != MATGEN_SUCCESS) {
    MATGEN_LOG_ERROR("Failed to sort COO matrix");
    matgen_coo_destroy(coo);
    return err;
  }

  err = matgen_coo_sum_duplicates_with_policy(coo, MATGEN_EXEC_PAR);
  if (err != MATGEN_SUCCESS) {
    MATGEN_LOG_ERROR("Failed to sum duplicates");
    matgen_coo_destroy(coo);
    return err;
  }

  MATGEN_LOG_DEBUG("After deduplication: %zu entries", coo->nnz);

  // Convert to CSR using OpenMP policy
  *result = matgen_coo_to_csr_with_policy(coo, MATGEN_EXEC_PAR);
  matgen_coo_destroy(coo);

  if (!(*result)) {
    MATGEN_LOG_ERROR("Failed to convert COO to CSR matrix");
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  MATGEN_LOG_DEBUG("Bilinear scaling (OMP) completed: output NNZ = %zu",
                   (*result)->nnz);

  return MATGEN_SUCCESS;
}
