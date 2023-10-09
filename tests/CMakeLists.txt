find_package(GTest REQUIRED CONFIG)
include(GoogleTest)

add_executable(${PROJECT_NAME}-tests criterions.cpp functions.cpp operations.cpp)
target_link_libraries(${PROJECT_NAME}-tests PRIVATE ${PROJECT_NAME} GTest::gtest GTest::gtest_main GTest::gmock)
target_compile_features(${PROJECT_NAME}-tests PRIVATE cxx_std_17)
gtest_add_tests(TARGET ${PROJECT_NAME}-tests)
