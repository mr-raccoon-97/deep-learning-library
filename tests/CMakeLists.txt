find_package(GTest REQUIRED CONFIG)
include(GoogleTest)

add_executable(${PROJECT_NAME}Tests criterions.cpp functions.cpp operations.cpp subscripts.cpp)
target_link_libraries(${PROJECT_NAME}Tests PRIVATE ${PROJECT_NAME} GTest::gtest GTest::gtest_main)
target_compile_options(${PROJECT_NAME}Tests PRIVATE cxx_std_20)
gtest_add_tests(TARGET ${PROJECT_NAME}Tests)
