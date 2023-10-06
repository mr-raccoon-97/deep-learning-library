cmake_minimum_required(VERSION 3.22)
project(cabernet-examples CXX)

if(PROJECT_IS_TOP_LEVEL)
    find_package(cabernet CONFIG REQUIRED)
endif()

function(add_example example)
    add_executable(${PROJECT_NAME}-${example} ${example}.cpp)
    target_link_libraries(${PROJECT_NAME}-${example} PRIVATE cabernet::cabernet)
    add_custom_command(
    TARGET ${PROJECT_NAME}-${example} POST_BUILD
    COMMAND $<TARGET_FILE:${PROJECT_NAME}-${example}>
    VERBATIM)
    target_compile_features(${PROJECT_NAME}-${example} PRIVATE cxx_std_17)
endfunction(add_example)

add_example(functions)
add_example(layers)
add_example(operations)
