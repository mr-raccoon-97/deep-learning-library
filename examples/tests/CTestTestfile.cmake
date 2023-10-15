# CMake generated Testfile for 
# Source directory: /home/eric/deep-learning-library/tests
# Build directory: /home/eric/deep-learning-library/examples/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[criterion.loss]=] "/home/eric/deep-learning-library/examples/tests/cabernetTests" "--gtest_filter=criterion.loss")
set_tests_properties([=[criterion.loss]=] PROPERTIES  _BACKTRACE_TRIPLES "/usr/share/cmake-3.22/Modules/GoogleTest.cmake;400;add_test;/home/eric/deep-learning-library/tests/CMakeLists.txt;7;gtest_add_tests;/home/eric/deep-learning-library/tests/CMakeLists.txt;0;")
add_test([=[functions.gradient]=] "/home/eric/deep-learning-library/examples/tests/cabernetTests" "--gtest_filter=functions.gradient")
set_tests_properties([=[functions.gradient]=] PROPERTIES  _BACKTRACE_TRIPLES "/usr/share/cmake-3.22/Modules/GoogleTest.cmake;400;add_test;/home/eric/deep-learning-library/tests/CMakeLists.txt;7;gtest_add_tests;/home/eric/deep-learning-library/tests/CMakeLists.txt;0;")
add_test([=[operations.matmul]=] "/home/eric/deep-learning-library/examples/tests/cabernetTests" "--gtest_filter=operations.matmul")
set_tests_properties([=[operations.matmul]=] PROPERTIES  _BACKTRACE_TRIPLES "/usr/share/cmake-3.22/Modules/GoogleTest.cmake;400;add_test;/home/eric/deep-learning-library/tests/CMakeLists.txt;7;gtest_add_tests;/home/eric/deep-learning-library/tests/CMakeLists.txt;0;")
add_test([=[optimizer.sgd]=] "/home/eric/deep-learning-library/examples/tests/cabernetTests" "--gtest_filter=optimizer.sgd")
set_tests_properties([=[optimizer.sgd]=] PROPERTIES  _BACKTRACE_TRIPLES "/usr/share/cmake-3.22/Modules/GoogleTest.cmake;400;add_test;/home/eric/deep-learning-library/tests/CMakeLists.txt;7;gtest_add_tests;/home/eric/deep-learning-library/tests/CMakeLists.txt;0;")
