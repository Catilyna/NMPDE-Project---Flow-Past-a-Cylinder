cmake_minimum_required(VERSION 3.12.0)
project(NavierStokesProject LANGUAGES CXX C)

# Set -O3 optimization by default for all build types
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")

# If using a custom deal.II module, help CMake find it
# You may need to adjust DEAL_II_DIR to match your module path
set(DEAL_II_DIR "/u/sw/toolchains/gcc-glibc/11.2.0/pkgs/dealii/9.5.1" CACHE PATH "Path to deal.II installation")
find_package(deal.II REQUIRED HINTS ${DEAL_II_DIR} NO_DEFAULT_PATH)

include(./common/cmake-common.cmake)

#add_executable(Stokes
#    src/test_stokes.cpp
#    src/Stokes.cpp
#    src/BoundaryValues.cpp
#)
#deal_ii_setup_target(Stokes)

add_executable(StationaryNavierStokes
    src/test_sns.cpp
    src/StationaryNavierStokes.cpp
    src/preconditioners/BlockSchurPreconditioner.cpp
    src/preconditioners/BlockTriangularPrecondition.cpp
)
deal_ii_setup_target(StationaryNavierStokes)

add_executable(NavierStokes
    src/test_ns.cpp
    src/NonStationaryNavierStokes.cpp
    src/preconditioners/BlockSchurPreconditioner.cpp
    src/preconditioners/BlockTriangularPrecondition.cpp
)
deal_ii_setup_target(NavierStokes)


