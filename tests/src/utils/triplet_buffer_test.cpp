#include <gtest/gtest.h>
#include <matgen/utils/triplet_buffer.h>

// =============================================================================
// Basic Functionality Tests
// =============================================================================

TEST(TripletBufferTest, CreateAndDestroy) {
  matgen_triplet_buffer_t* buffer = matgen_triplet_buffer_create(10);
  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(matgen_triplet_buffer_size(buffer), 0);
  EXPECT_GE(matgen_triplet_buffer_capacity(buffer), 10);
  matgen_triplet_buffer_destroy(buffer);
}

TEST(TripletBufferTest, CreateWithZeroCapacity) {
  // Should use default capacity
  matgen_triplet_buffer_t* buffer = matgen_triplet_buffer_create(0);
  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(matgen_triplet_buffer_size(buffer), 0);
  EXPECT_GT(matgen_triplet_buffer_capacity(buffer), 0);
  matgen_triplet_buffer_destroy(buffer);
}

TEST(TripletBufferTest, DestroyNull) {
  // Should not crash
  matgen_triplet_buffer_destroy(nullptr);
}

TEST(TripletBufferTest, SizeAndCapacityNull) {
  EXPECT_EQ(matgen_triplet_buffer_size(nullptr), 0);
  EXPECT_EQ(matgen_triplet_buffer_capacity(nullptr), 0);
}

// =============================================================================
// Adding Entries Tests
// =============================================================================

TEST(TripletBufferTest, AddSingleEntry) {
  matgen_triplet_buffer_t* buffer = matgen_triplet_buffer_create(10);
  ASSERT_NE(buffer, nullptr);

  matgen_error_t err = matgen_triplet_buffer_add(buffer, 5, 3, 2.5);
  EXPECT_EQ(err, MATGEN_SUCCESS);
  EXPECT_EQ(matgen_triplet_buffer_size(buffer), 1);

  // Verify the entry was added
  EXPECT_EQ(buffer->rows[0], 5);
  EXPECT_EQ(buffer->cols[0], 3);
  EXPECT_DOUBLE_EQ(buffer->vals[0], 2.5);

  matgen_triplet_buffer_destroy(buffer);
}

TEST(TripletBufferTest, AddMultipleEntries) {
  matgen_triplet_buffer_t* buffer = matgen_triplet_buffer_create(10);
  ASSERT_NE(buffer, nullptr);

  // Add several entries
  EXPECT_EQ(matgen_triplet_buffer_add(buffer, 0, 0, 1.0), MATGEN_SUCCESS);
  EXPECT_EQ(matgen_triplet_buffer_add(buffer, 1, 2, 3.5), MATGEN_SUCCESS);
  EXPECT_EQ(matgen_triplet_buffer_add(buffer, 5, 5, 7.2), MATGEN_SUCCESS);

  EXPECT_EQ(matgen_triplet_buffer_size(buffer), 3);

  // Verify all entries
  EXPECT_EQ(buffer->rows[0], 0);
  EXPECT_EQ(buffer->cols[0], 0);
  EXPECT_DOUBLE_EQ(buffer->vals[0], 1.0);

  EXPECT_EQ(buffer->rows[1], 1);
  EXPECT_EQ(buffer->cols[1], 2);
  EXPECT_DOUBLE_EQ(buffer->vals[1], 3.5);

  EXPECT_EQ(buffer->rows[2], 5);
  EXPECT_EQ(buffer->cols[2], 5);
  EXPECT_DOUBLE_EQ(buffer->vals[2], 7.2);

  matgen_triplet_buffer_destroy(buffer);
}

TEST(TripletBufferTest, AddWithNull) {
  matgen_error_t err = matgen_triplet_buffer_add(nullptr, 0, 0, 1.0);
  EXPECT_EQ(err, MATGEN_ERROR_INVALID_ARGUMENT);
}

// =============================================================================
// Automatic Resizing Tests
// =============================================================================

TEST(TripletBufferTest, AutomaticResize) {
  // Create with small capacity
  matgen_triplet_buffer_t* buffer = matgen_triplet_buffer_create(2);
  ASSERT_NE(buffer, nullptr);

  size_t initial_capacity = matgen_triplet_buffer_capacity(buffer);
  EXPECT_EQ(initial_capacity, 2);

  // Add entries to trigger resize
  EXPECT_EQ(matgen_triplet_buffer_add(buffer, 0, 0, 1.0), MATGEN_SUCCESS);
  EXPECT_EQ(matgen_triplet_buffer_add(buffer, 1, 1, 2.0), MATGEN_SUCCESS);
  EXPECT_EQ(matgen_triplet_buffer_add(buffer, 2, 2, 3.0),
            MATGEN_SUCCESS);  // Triggers resize

  EXPECT_EQ(matgen_triplet_buffer_size(buffer), 3);
  EXPECT_GT(matgen_triplet_buffer_capacity(buffer), initial_capacity);

  // Verify all data is still correct after resize
  EXPECT_EQ(buffer->rows[0], 0);
  EXPECT_DOUBLE_EQ(buffer->vals[0], 1.0);
  EXPECT_EQ(buffer->rows[1], 1);
  EXPECT_DOUBLE_EQ(buffer->vals[1], 2.0);
  EXPECT_EQ(buffer->rows[2], 2);
  EXPECT_DOUBLE_EQ(buffer->vals[2], 3.0);

  matgen_triplet_buffer_destroy(buffer);
}

TEST(TripletBufferTest, MultipleResizes) {
  // Start very small to force multiple resizes
  matgen_triplet_buffer_t* buffer = matgen_triplet_buffer_create(1);
  ASSERT_NE(buffer, nullptr);

  // Add many entries to trigger multiple resizes
  const size_t num_entries = 100;
  for (size_t i = 0; i < num_entries; i++) {
    matgen_error_t err =
        matgen_triplet_buffer_add(buffer, i, i, (matgen_value_t)i);
    EXPECT_EQ(err, MATGEN_SUCCESS);
  }

  EXPECT_EQ(matgen_triplet_buffer_size(buffer), num_entries);

  // Verify all entries are correct
  for (size_t i = 0; i < num_entries; i++) {
    EXPECT_EQ(buffer->rows[i], i);
    EXPECT_EQ(buffer->cols[i], i);
    EXPECT_DOUBLE_EQ(buffer->vals[i], (matgen_value_t)i);
  }

  matgen_triplet_buffer_destroy(buffer);
}

// =============================================================================
// Clear Tests
// =============================================================================

TEST(TripletBufferTest, Clear) {
  matgen_triplet_buffer_t* buffer = matgen_triplet_buffer_create(10);
  ASSERT_NE(buffer, nullptr);

  // Add some entries
  matgen_triplet_buffer_add(buffer, 0, 0, 1.0);
  matgen_triplet_buffer_add(buffer, 1, 1, 2.0);
  matgen_triplet_buffer_add(buffer, 2, 2, 3.0);

  EXPECT_EQ(matgen_triplet_buffer_size(buffer), 3);
  size_t capacity = matgen_triplet_buffer_capacity(buffer);

  // Clear the buffer
  matgen_triplet_buffer_clear(buffer);

  EXPECT_EQ(matgen_triplet_buffer_size(buffer), 0);
  EXPECT_EQ(matgen_triplet_buffer_capacity(buffer),
            capacity);  // Capacity unchanged

  // Should be able to add again
  EXPECT_EQ(matgen_triplet_buffer_add(buffer, 5, 5, 5.0), MATGEN_SUCCESS);
  EXPECT_EQ(matgen_triplet_buffer_size(buffer), 1);

  matgen_triplet_buffer_destroy(buffer);
}

TEST(TripletBufferTest, ClearNull) {
  // Should not crash
  matgen_triplet_buffer_clear(nullptr);
}

// =============================================================================
// Edge Cases and Stress Tests
// =============================================================================

TEST(TripletBufferTest, ZeroValues) {
  matgen_triplet_buffer_t* buffer = matgen_triplet_buffer_create(10);
  ASSERT_NE(buffer, nullptr);

  // Zero values should be stored
  EXPECT_EQ(matgen_triplet_buffer_add(buffer, 0, 0, 0.0), MATGEN_SUCCESS);
  EXPECT_EQ(matgen_triplet_buffer_size(buffer), 1);
  EXPECT_DOUBLE_EQ(buffer->vals[0], 0.0);

  matgen_triplet_buffer_destroy(buffer);
}

TEST(TripletBufferTest, NegativeValues) {
  matgen_triplet_buffer_t* buffer = matgen_triplet_buffer_create(10);
  ASSERT_NE(buffer, nullptr);

  EXPECT_EQ(matgen_triplet_buffer_add(buffer, 1, 2, -3.5), MATGEN_SUCCESS);
  EXPECT_EQ(matgen_triplet_buffer_size(buffer), 1);
  EXPECT_DOUBLE_EQ(buffer->vals[0], -3.5);

  matgen_triplet_buffer_destroy(buffer);
}

TEST(TripletBufferTest, LargeIndices) {
  matgen_triplet_buffer_t* buffer = matgen_triplet_buffer_create(10);
  ASSERT_NE(buffer, nullptr);

  matgen_index_t large_idx = 1000000;
  EXPECT_EQ(matgen_triplet_buffer_add(buffer, large_idx, large_idx, 1.0),
            MATGEN_SUCCESS);
  EXPECT_EQ(buffer->rows[0], large_idx);
  EXPECT_EQ(buffer->cols[0], large_idx);

  matgen_triplet_buffer_destroy(buffer);
}

TEST(TripletBufferTest, DuplicateCoordinates) {
  // Buffer should allow duplicates (they'll be summed later by COO)
  matgen_triplet_buffer_t* buffer = matgen_triplet_buffer_create(10);
  ASSERT_NE(buffer, nullptr);

  EXPECT_EQ(matgen_triplet_buffer_add(buffer, 1, 2, 3.0), MATGEN_SUCCESS);
  EXPECT_EQ(matgen_triplet_buffer_add(buffer, 1, 2, 4.0), MATGEN_SUCCESS);
  EXPECT_EQ(matgen_triplet_buffer_add(buffer, 1, 2, 5.0), MATGEN_SUCCESS);

  EXPECT_EQ(matgen_triplet_buffer_size(buffer), 3);

  // All three should be stored
  EXPECT_EQ(buffer->rows[0], 1);
  EXPECT_EQ(buffer->cols[0], 2);
  EXPECT_DOUBLE_EQ(buffer->vals[0], 3.0);

  EXPECT_EQ(buffer->rows[1], 1);
  EXPECT_EQ(buffer->cols[1], 2);
  EXPECT_DOUBLE_EQ(buffer->vals[1], 4.0);

  EXPECT_EQ(buffer->rows[2], 1);
  EXPECT_EQ(buffer->cols[2], 2);
  EXPECT_DOUBLE_EQ(buffer->vals[2], 5.0);

  matgen_triplet_buffer_destroy(buffer);
}

TEST(TripletBufferTest, LargeBuffer) {
  // Test with a larger number of entries
  const size_t num_entries = 10000;
  matgen_triplet_buffer_t* buffer = matgen_triplet_buffer_create(100);
  ASSERT_NE(buffer, nullptr);

  // Add many entries
  for (size_t i = 0; i < num_entries; i++) {
    matgen_index_t row = i / 100;
    matgen_index_t col = i % 100;
    matgen_value_t val = (matgen_value_t)((double)i * 0.1);

    EXPECT_EQ(matgen_triplet_buffer_add(buffer, row, col, val), MATGEN_SUCCESS);
  }

  EXPECT_EQ(matgen_triplet_buffer_size(buffer), num_entries);

  // Spot check some values
  EXPECT_EQ(buffer->rows[0], 0);
  EXPECT_EQ(buffer->cols[0], 0);
  EXPECT_DOUBLE_EQ(buffer->vals[0], 0.0);

  EXPECT_EQ(buffer->rows[num_entries - 1], (num_entries - 1) / 100);
  EXPECT_EQ(buffer->cols[num_entries - 1], (num_entries - 1) % 100);
  EXPECT_DOUBLE_EQ(buffer->vals[num_entries - 1],
                   (matgen_value_t)((num_entries - 1) * 0.1));

  matgen_triplet_buffer_destroy(buffer);
}

TEST(TripletBufferTest, ClearAndReuse) {
  matgen_triplet_buffer_t* buffer = matgen_triplet_buffer_create(5);
  ASSERT_NE(buffer, nullptr);

  // Add entries, clear, add again multiple times
  for (int cycle = 0; cycle < 3; cycle++) {
    matgen_triplet_buffer_clear(buffer);
    EXPECT_EQ(matgen_triplet_buffer_size(buffer), 0);

    for (int i = 0; i < 10; i++) {
      EXPECT_EQ(matgen_triplet_buffer_add(buffer, i, i, (matgen_value_t)i),
                MATGEN_SUCCESS);
    }

    EXPECT_EQ(matgen_triplet_buffer_size(buffer), 10);
  }

  matgen_triplet_buffer_destroy(buffer);
}
