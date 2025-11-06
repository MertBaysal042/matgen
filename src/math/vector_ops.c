#include "matgen/math/vector_ops.h"

#include <math.h>
#include <string.h>

// =============================================================================
// Norms and Dot Products
// =============================================================================

matgen_value_t matgen_vec_dot(const matgen_value_t* x, const matgen_value_t* y,
                              matgen_size_t n) {
  matgen_value_t result = 0.0;
  for (matgen_size_t i = 0; i < n; i++) {
    result += x[i] * y[i];
  }
  return result;
}

matgen_value_t matgen_vec_norm2(const matgen_value_t* x, matgen_size_t n) {
  matgen_value_t sum = 0.0;
  for (matgen_size_t i = 0; i < n; i++) {
    sum += x[i] * x[i];
  }
  return sqrt(sum);
}

matgen_value_t matgen_vec_norm1(const matgen_value_t* x, matgen_size_t n) {
  matgen_value_t sum = 0.0;
  for (matgen_size_t i = 0; i < n; i++) {
    sum += fabs(x[i]);
  }
  return sum;
}

matgen_value_t matgen_vec_norminf(const matgen_value_t* x, matgen_size_t n) {
  matgen_value_t max_val = 0.0;
  for (matgen_size_t i = 0; i < n; i++) {
    matgen_value_t abs_val = fabs(x[i]);
    if (abs_val > max_val) {
      max_val = abs_val;
    }
  }
  return max_val;
}

// =============================================================================
// Basic Vector Operations
// =============================================================================

void matgen_vec_scale(matgen_value_t alpha, const matgen_value_t* x,
                      matgen_value_t* y, matgen_size_t n) {
  for (matgen_size_t i = 0; i < n; i++) {
    y[i] = alpha * x[i];
  }
}

void matgen_vec_add(const matgen_value_t* x, const matgen_value_t* y,
                    matgen_value_t* z, matgen_size_t n) {
  for (matgen_size_t i = 0; i < n; i++) {
    z[i] = x[i] + y[i];
  }
}

void matgen_vec_sub(const matgen_value_t* x, const matgen_value_t* y,
                    matgen_value_t* z, matgen_size_t n) {
  for (matgen_size_t i = 0; i < n; i++) {
    z[i] = x[i] - y[i];
  }
}

void matgen_vec_axpy(matgen_value_t alpha, const matgen_value_t* x,
                     matgen_value_t* y, matgen_size_t n) {
  for (matgen_size_t i = 0; i < n; i++) {
    y[i] += alpha * x[i];
  }
}

void matgen_vec_axpby(matgen_value_t alpha, const matgen_value_t* x,
                      matgen_value_t beta, const matgen_value_t* y,
                      matgen_value_t* z, matgen_size_t n) {
  for (matgen_size_t i = 0; i < n; i++) {
    z[i] = (alpha * x[i]) + (beta * y[i]);
  }
}

// =============================================================================
// Utility Operations
// =============================================================================

void matgen_vec_copy(const matgen_value_t* x, matgen_value_t* y,
                     matgen_size_t n) {
  memcpy(y, x, n * sizeof(matgen_value_t));
}

void matgen_vec_zero(matgen_value_t* x, matgen_size_t n) {
  memset(x, 0, n * sizeof(matgen_value_t));
}

void matgen_vec_fill(matgen_value_t* x, matgen_value_t alpha, matgen_size_t n) {
  for (matgen_size_t i = 0; i < n; i++) {
    x[i] = alpha;
  }
}
