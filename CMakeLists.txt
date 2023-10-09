cmake_minimum_required(VERSION 3.22)
project(CaberNet)
set(CMAKE_CXX_STANDARD 17)

# Define source and header file variables

set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(HEADERS
    ${INCLUDE_DIR}/CaberNet.h
    ${INCLUDE_DIR}/CaberNet/tensor.h
    ${INCLUDE_DIR}/CaberNet/tensor/tensor_float.h
    ${INCLUDE_DIR}/CaberNet/tensor/tensor_int.h
    ${INCLUDE_DIR}/CaberNet/functions.h
    ${INCLUDE_DIR}/CaberNet/layers.h
    ${INCLUDE_DIR}/CaberNet/model.h
    ${INCLUDE_DIR}/CaberNet/optimizers.h
    ${INCLUDE_DIR}/CaberNet/criterions.h
)

set(SOURCES
    ${SOURCE_DIR}/tensor.cpp
    ${SOURCE_DIR}/tensor/tensor_float.cpp
    ${SOURCE_DIR}/tensor/tensor_int.cpp
    ${SOURCE_DIR}/functions.cpp
    ${SOURCE_DIR}/layers.cpp
    ${SOURCE_DIR}/criterions.cpp
    ${SOURCE_DIR}/optimizers.cpp
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
target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen) # TODO(prince-chrismc): Is this actually private?


# Install the library and header files
install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)

install(DIRECTORY ${INCLUDE_DIR}/
        DESTINATION include/${PROJECT_NAME}
        FILES_MATCHING PATTERN "*.h"
)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
