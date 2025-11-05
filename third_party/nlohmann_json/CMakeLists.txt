message(STATUS "Fetching nlohmann_json...")

FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.12.0
  GIT_SHALLOW    TRUE
  SYSTEM
  EXCLUDE_FROM_ALL
)

set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(json)

message(STATUS "nlohmann_json fetched successfully")
