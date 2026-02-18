# NAVIER STOKES SOLVER (NMPDE Project)

**Overview**

This repository contains a C++ finite element solver for the incompressible Navier-Stokes equations, developed as a project for the Numerical Methods for Partial Differential Equations course.

The project is built on top of the open-source deal.II finite element library and provides two distinct numerical approaches for solving the fluid dynamics equations: a classical fully coupled monolithic solver, and a fractional-step projection method.

### Dependencies

To compile and run this project, you will need the following installed on your system:

- **C++ Compiler**: GCC/G++

- **deal.II**: Finite element library (with MPI and Trilinos support)

- **CMake**: Version 3.10 or higher

- **Python 3**: With venv support (for automated mesh generation)

- **ParaView**: For visualizing output files

### Instructions

This project automatically builds all the necessary files for solvers execution. In particular, the CMake build system creates a Python virtual environment and install the required packages for mesh generation.

**Step by step guide to the execution:**

- Clone the repository and navigate into it:

```bash
git clone https://github.com/Catilyna/NMPDE-Project---Flow-Past-a-Cylinder.git
cd NPDE-Project---Flow-Past-a-Cylinder
```

- Load necessary modules:

```bash
$ module load gcc-glibc dealii
```

- Run the following preliminary commands:

```bash
mkdir build
cd build
cmake ..
```

- Compile the executables and create meshes with a single command inside the `build` directory:

```bash
make
```

Meshes will be generated inside the `./mesh` directory, while executables will be created into `./build`.

### Execution:

- First 2D test commands execution:

```bash

```

- Second 2D test commands execution:

```bash

```

- Third 2D test commands execution:

```bash

```

- First 3D test commands execution:

```bash

```

- Second 3D test commands execution:

```bash
./NavierStokes -d 3 -f ../mesh/3D/mesh3D_coarse_cylinder.msh -v 0.001 -u 2.25 -T 4.0 -dt 0.2 -td false
```

- Third 3D test commands execution:

```bash
./NavierStokes -d 3 -f ../mesh/3D/mesh3D_coarse_cylinder.msh -v 0.001 -u 2.25 -T 4.0 -dt 0.2 -td true
```

- Example test in order to show Chorin Temam potential with high Reynolds numbers:

(Luca inserisci i dati sulla simulazione da te fatta)

```bash
./NavierStokesChorinTemam
```

**Some usefull remarks:**

- type the following command to get any help on the flags and their usage:

```bash
./NavierStokes -h
```

- Each of these commands can be run in parallel using `Open_MPI` command `mpirun` followed by the flag `-n n_proc` in order to execute the scripts in parallel and exploit the Trilinos module features.

- Different meshes file can be found in `./mesh` directory, such as finer meshes and the 'parallelepiped` ones (just for 3D tests).

- Results are stored in a `./results/common` folder, ready to be visualized using Paraview visualization software.
