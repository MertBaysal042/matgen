target_sources(
  matgen
  PRIVATE
    # sources
)

set_source_files_properties(
  # sources
  PROPERTIES
    LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
)
