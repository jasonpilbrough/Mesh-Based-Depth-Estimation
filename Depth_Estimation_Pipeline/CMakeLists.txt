cmake_minimum_required(VERSION 3.10)

# set the project name
project(Sandbox VERSION 1.0 LANGUAGES CXX)


# If you set any CMAKE_ variables, that can go here:
# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "-O3") #-Wall -O3 

# Let's ensure -std=c++xx instead of -std=g++xx
set(CMAKE_CXX_EXTENSIONS OFF)



### Require out-of-source builds
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
    message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()



# Find Required libraries
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)
#find_package (Eigen3 3.3 NO_MODULE)

#find_library (
#    libsparsestereo
#    NAMES libsparsestereo.a # what to look for
#    HINTS "${CMAKE_SOURCE_DIR}/lib" # where to look
#    NO_DEFAULT_PATH # do not search system default paths
#)

# check if we found the library
#message(STATUS "SPARSE_STEREO_LIB: [${SPARSE_STEREO_LIB}]")

# Project includes
include_directories(SYSTEM src
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS})
#  ${EIGEN3_INCLUDE_DIRS})
#  ${Sophus_INCLUDE_DIRS})


# The compiled library code is here
add_subdirectory(src)

# The executable code is here
add_subdirectory(app)

#set the default path for built libraries to the "lib" directory
#set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)





# Add the executable - Output name matches target name
#add_executable(${PROJECT_NAME} sandbox.cpp)

#add include directory
#target_include_directories(${PROJECT_NAME} PUBLIC include)

#target_include_directories(include PUBLIC "${PROJECT_BINARY_DIR}")
