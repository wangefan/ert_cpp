cmake_minimum_required(VERSION 2.8)

# project name
project(TestVideoMain)

# include
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/ERT)
include_directories( ${OpenCV_INCLUDE_DIRS} )

# set source
set(source ${CMAKE_SOURCE_DIR}/ERT/Configuration.cpp ${CMAKE_SOURCE_DIR}/ERT/ERT.cpp ${CMAKE_SOURCE_DIR}/ERT/Regressor.cpp ${CMAKE_SOURCE_DIR}/ERT/Ferm.cpp ${CMAKE_SOURCE_DIR}/ERT/Utils.cpp ${CMAKE_SOURCE_DIR}/ERT/SampleData.cpp)

# add exe
add_executable(TestVideoMain TestVideoMain.cpp  ${source})

# package openCV
find_package(OpenCV REQUIRED)
target_link_libraries(TestVideoMain ${OpenCV_LIBS})

# package JSON
include(FetchContent)
FetchContent_Declare(json
  GIT_REPOSITORY https://github.com/nlohmann/json
  GIT_TAG v3.7.3)
FetchContent_GetProperties(json)
if(NOT json_POPULATED)
  FetchContent_Populate(json)
  add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
target_link_libraries(TestVideoMain PRIVATE nlohmann_json::nlohmann_json)
