cmake_minimum_required(VERSION 3.22)
project(cabernet-examples CXX)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    # This is the top level CMakeLists.txt
    find_package(cabernet CONFIG REQUIRED)
endif()

function(add_example example)
    add_executable(${CMAKE_PROJECT_NAME}-${example} ${example}.cpp)
    target_link_libraries(${CMAKE_PROJECT_NAME}-${example} PRIVATE cabernet::cabernet)
    add_custom_command(
    TARGET ${CMAKE_PROJECT_NAME}-${example} POST_BUILD
    COMMAND $<TARGET_FILE:${CMAKE_PROJECT_NAME}-${example}>
    VERBATIM)
endfunction(add_example)

add_example(functions)
add_example(layers)
add_example(operations)
