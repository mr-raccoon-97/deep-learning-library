cmake_minimum_required(VERSION 3.22)
project(cabernet-examples)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    # This is the top level CMakeLists.txt
    find_package(cabernet CONFIG REQUIRED)
endif()

add_executable(${PROJECT_NAME}-functions functions.cpp)
target_link_libraries(${PROJECT_NAME}-functions PRIVATE ${PROJECT_NAME})

add_executable(${PROJECT_NAME}-layers layers.cpp)
target_link_libraries(${PROJECT_NAME}-layers PRIVATE ${PROJECT_NAME})

add_executable(${PROJECT_NAME}-operations operations.cpp)
target_link_libraries(${PROJECT_NAME}-operations PRIVATE ${PROJECT_NAME})