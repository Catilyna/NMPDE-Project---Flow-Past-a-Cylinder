cmake_minimum_required(VERSION 3.12.0)
project(NavierStokesProject LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(NavierStokes
	src/main.cpp
	src/StationaryNavierStokes.cpp
	src/BoundaryValues.cpp
	src/BlockSchurPreconditioner.cpp
)
deal_ii_setup_target(NavierStokes)
