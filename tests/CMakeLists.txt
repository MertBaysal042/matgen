add_executable(matgen_test)

add_subdirectory("src")

target_link_libraries(
  matgen_test
  PRIVATE
    matgen
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(matgen_test)
