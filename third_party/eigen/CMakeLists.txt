message(STATUS "Fetching Eigen...")

FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.1
  GIT_SHALLOW    TRUE
  SYSTEM
  EXCLUDE_FROM_ALL
)

set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(eigen)

message(STATUS "Eigen fetched successfully")
