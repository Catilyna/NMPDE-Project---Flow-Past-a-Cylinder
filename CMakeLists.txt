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
include_directories("${PROJECT_SOURCE_DIR}/include")

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(VENV_DIR "${CMAKE_CURRENT_BINARY_DIR}/.venv")
set(REQ_FILE "${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt")

if(WIN32)
    set(VENV_PYTHON "${VENV_DIR}/Scripts/python.exe")
else()
    set(VENV_PYTHON "${VENV_DIR}/bin/python")
endif()

# 3. Define a command to create the venv and install dependencies.
# This ONLY runs if the venv doesn't exist or if requirements.txt changes.
add_custom_command(
    OUTPUT "${VENV_DIR}/.created"
    COMMAND Python3::Interpreter -m venv "${VENV_DIR}"
    COMMAND "${VENV_PYTHON}" -m pip install --upgrade pip
    COMMAND "${VENV_PYTHON}" -m pip install -r "${REQ_FILE}"
    COMMAND ${CMAKE_COMMAND} -E touch "${VENV_DIR}/.created"
    DEPENDS "${REQ_FILE}"
)

add_custom_target(generate_meshes ALL
    COMMAND "${VENV_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/mesh_gen/Mesh2D.py"
    COMMAND "${VENV_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/mesh_gen/Mesh3D.py"
    DEPENDS "${VENV_DIR}/.created"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

#add_executable(Stokes
#    src/test_stokes.cpp
#    src/Stokes.cpp
#    src/BoundaryValues.cpp
#)
#deal_ii_setup_target(Stokes)

#add_executable(StationaryNavierStokes
#    src/test_sns.cpp
#    src/StationaryNavierStokes.cpp
#    src/preconditioners/BlockSchurPreconditioner.cpp
#    src/preconditioners/BlockTriangularPrecondition.cpp
#)
#deal_ii_setup_target(StationaryNavierStokes)


add_executable(NavierStokes
    src/test_ns.cpp
    src/NonStationaryNavierStokes.cpp
    src/preconditioners/BlockSchurPreconditioner.cpp
    src/preconditioners/BlockTriangularPrecondition.cpp
)
deal_ii_setup_target(NavierStokes)

add_executable(NavierStokesChorinTemam
    src/test_fns.cpp
    src/NavierStokesFractional.cpp
    src/NonStationaryNavierStokes.cpp
    src/preconditioners/BlockSchurPreconditioner.cpp
    src/preconditioners/BlockTriangularPrecondition.cpp
)
deal_ii_setup_target(NavierStokesChorinTemam)



