if(MATGEN_ENABLE_OPENMP)
  add_subdirectory("omp")
endif()

if(MATGEN_ENABLE_MPI)
  add_subdirectory("mpi")
endif()

if(MATGEN_ENABLE_CUDA)
  add_subdirectory("cuda")
endif()
