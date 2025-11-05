include(FetchContent)

set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

message(STATUS "Fetching third-party dependencies...")

if(MATGEN_BUILD_TESTS)
  add_subdirectory("googletest")
endif()

if(MATGEN_BUILD_BENCHMARKS)
  add_subdirectory("benchmark")
endif()
