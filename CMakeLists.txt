cmake_minimum_required(VERSION 3.12.0)
project(NavierStokesProject LANGUAGES CXX C)

# If using a custom deal.II module, help CMake find it
# You may need to adjust DEAL_II_DIR to match your module path
set(DEAL_II_DIR "/u/sw/toolchains/gcc-glibc/11.2.0/pkgs/dealii/9.5.1" CACHE PATH "Path to deal.II installation")
find_package(deal.II REQUIRED HINTS ${DEAL_II_DIR} NO_DEFAULT_PATH)

include(./common/cmake-common.cmake)

add_executable(Stokes
    src/main.cpp
    src/Stokes.cpp
    src/BoundaryValues.cpp
)

deal_ii_setup_target(NavierStokes)
