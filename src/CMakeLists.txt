target_sources(
  matgen
  PRIVATE
    # sources
)

add_subdirectory("core")
add_subdirectory("util")

if(MATGEN_ENABLE_OPENMP OR MATGEN_ENABLE_MPI OR MATGEN_ENABLE_CUDA)
  add_subdirectory("parallel")
endif()
