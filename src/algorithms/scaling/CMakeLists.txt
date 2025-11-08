target_sources(
  matgen
  PRIVATE
    "bilinear.c"
    "nearest_neighbor.c"
)

if(MATGEN_ENABLE_OPENMP)
  target_sources(
    matgen
    PRIVATE
      "bilinear_omp.c"
  )
endif()
