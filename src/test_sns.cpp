#include <iostream>
#include "StationaryNavierStokes.hpp"

int main(int argc, char* argv[])
{
    using namespace NavierStokes;
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
    const std::string  mesh_file_name  = "../mesh/mesh3D_example.msh";
    const unsigned int degree_velocity = 2;
    const unsigned int degree_pressure = 1;

    try {
            StationaryNavierStokes<3> flow(mesh_file_name, degree_velocity, degree_pressure);
            flow.run();
            return 0;
        } catch (std::exception &exc) {
            std::cerr << std::endl << std::endl
                      << "----------------------------------------------------" << std::endl;
            std::cerr << "Exception on processing: " << std::endl
                      << exc.what() << std::endl
                      << "Aborting!" << std::endl
                      << "----------------------------------------------------" << std::endl;
            return 1;
        }
}