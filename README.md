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
module load gcc-glibc dealii
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

Meshes will be generated inside the `./mesh` directory, while executables will be created into `./build`. Moreover, a folder `./results/common` will be created for storing the output of the results.

### Execution:

- First 2D test commands execution:

```bash
./NavierStokes -d 2 -f ../mesh/2D/mesh2D_coarse_cylinder.msh -v 0.001 -u 0.3 -T 5.0 -dt 0.02 -td false
```

- Second 2D test commands execution:

```bash
./NavierStokes -d 2 -f ../mesh/2D/mesh2D_coarse_cylinder.msh -v 0.001 -u 1.5 -T 5.0 -dt 0.02 -td false
```

- Third 2D test commands execution:

```bash
./NavierStokes -d 2 -f ../mesh/2D/mesh2D_coarse_cylinder.msh -v 0.001 -u 1.5 -T 8.0 -dt 0.02 -td true
```

- First 3D test commands execution:

```bash
./NavierStokes -d 3 -f ../mesh/3D/mesh3D_coarse_cylinder.msh -v 0.001 -u 0.45 -T 4.0 -dt 0.02 -td false
```

- Second 3D test commands execution:

```bash
./NavierStokes -d 3 -f ../mesh/3D/mesh3D_coarse_cylinder.msh -v 0.001 -u 2.25 -T 4.0 -dt 0.02 -td false
```

- Third 3D test commands execution:

```bash
./NavierStokes -d 3 -f ../mesh/3D/mesh3D_coarse_cylinder.msh -v 0.001 -u 2.25 -T 4.0 -dt 0.02 -td true
```

- Example test in order to show Chorin Temam potential with high Reynolds numbers:

```bash
./NavierStokesChorinTemam -d 2 -f ../mesh/2D/mesh2D_fine_cylinder.msh -v 0.001 -u 15.0 -T 5.0 -dt 0.001 -td false
```

**Some usefull remarks:**

- type the following command to get any help on the flags and their usage:

```bash
./NavierStokes -h
```

- Each of these commands can be run in parallel using `Open_MPI` command `mpirun` followed by the flag `-n n_proc` in order to execute the scripts in parallel and exploit the Trilinos module features (this is indeed HIGHLY recommended!).

- Different meshes file can be found in `./mesh` directory, such as finer meshes and the 'parallelepiped` ones (just for 3D tests).

- Results are stored in a `./results/common` folder, ready to be visualized using Paraview visualization software.

- In the `./results` folder a `drag_lift_history.txt` dataset is created in which to save the drag and lift coefficients at each timestep. To generate a plot, it is sufficient to run the Python script `plot_coefficients.py` that can be found in the `./plot_gen` folder.

- The `./report` folder contains a report of the projects, as well as the source code to generate such a report.
