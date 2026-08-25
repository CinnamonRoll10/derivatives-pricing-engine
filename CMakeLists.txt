cmake_minimum_required(VERSION 3.16)
project(derivatives_pricing_engine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

add_library(dpe
  src/analytic_bs.cpp
  src/binomial.cpp
  src/monte_carlo.cpp
  src/finite_difference.cpp
  src/implied_vol.cpp
)
target_include_directories(dpe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(dpe PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()

add_executable(dpe_tests tests/test_main.cpp)
target_link_libraries(dpe_tests PRIVATE dpe)
add_test(NAME dpe_tests COMMAND dpe_tests)

add_executable(convergence_study apps/convergence_study.cpp)
target_link_libraries(convergence_study PRIVATE dpe)
