cmake_minimum_required(VERSION 3.22)
project(cabernet CXX)

include(GNUInstallDirs) # Reasonable Defaults

option(CABERNET_BUILD_EXAMPLES "Build and run the examples demonstration important features" OFF)

# Define source and header file variables
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(HEADERS
    ${INCLUDE_DIR}/CaberNet.h
    ${INCLUDE_DIR}/CaberNet/tensor.h
    ${INCLUDE_DIR}/CaberNet/functions.h
    ${INCLUDE_DIR}/CaberNet/layers.h
    ${INCLUDE_DIR}/CaberNet/model.h
    ${INCLUDE_DIR}/CaberNet/subscripts.h
    ${INCLUDE_DIR}/CaberNet/criterions.h
)

set(SOURCES
    ${SOURCE_DIR}/tensor.cpp
    ${SOURCE_DIR}/functions.cpp
    ${SOURCE_DIR}/layers.cpp
    ${SOURCE_DIR}/subscripts.cpp
    ${SOURCE_DIR}/criterions.cpp
    ${SOURCE_DIR}/internals/operations/internal_operation_addition.cpp
    ${SOURCE_DIR}/internals/operations/internal_operation_multiplication.cpp
    ${SOURCE_DIR}/internals/operations/internal_operation_matmul.cpp
    ${SOURCE_DIR}/internals/functions/internal_function_linear.cpp
    ${SOURCE_DIR}/internals/functions/internal_function_softmax.cpp
    ${SOURCE_DIR}/internals/functions/internal_function_logsoftmax.cpp
    ${SOURCE_DIR}/internals/functions/internal_function_relu.cpp
    ${SOURCE_DIR}/internals/criterions/internal_criterion_nllloss.cpp
)

find_package(Eigen3 REQUIRED CONFIG)

# Create the library target
add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${INCLUDE_DIR}>  
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

# Install the library and header files
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
add_subdirectory(cmake)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

if(CABERNET_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()