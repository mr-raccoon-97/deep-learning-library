cmake_minimum_required(VERSION 3.22)
project(CaberNet)
set(CMAKE_CXX_STANDARD 17)

# Define source and header file variables

set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(HEADERS
    ${INCLUDE_DIR}/CaberNet.h
    ${INCLUDE_DIR}/CaberNet/tensor.h
    ${INCLUDE_DIR}/CaberNet/functions.h
    ${INCLUDE_DIR}/CaberNet/layers.h
    ${INCLUDE_DIR}/CaberNet/model.h
)

set(SOURCES
    ${SOURCE_DIR}/tensor.cpp
    ${SOURCE_DIR}/functions.cpp
    ${SOURCE_DIR}/layers.cpp
    ${SOURCE_DIR}/internals/internal_array_operations.cpp
    ${SOURCE_DIR}/internals/operations/internal_operation_addition.cpp
    ${SOURCE_DIR}/internals/operations/internal_operation_multiplication.cpp
    ${SOURCE_DIR}/internals/operations/internal_operation_matmul.cpp
    ${SOURCE_DIR}/internals/functions/internal_function_linear.cpp
    ${SOURCE_DIR}/internals/functions/internal_function_softmax.cpp
    ${SOURCE_DIR}/internals/functions/internal_function_logsoftmax.cpp
    ${SOURCE_DIR}/internals/functions/internal_function_relu.cpp
)


# Create the library target
add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Specify the public include directories for the library
target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDE_DIR})

# Set the output directory for the compiled library
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# Install the library and header files
install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)

install(DIRECTORY ${INCLUDE_DIR}/
        DESTINATION include/${PROJECT_NAME}
        FILES_MATCHING PATTERN "*.h"
)
