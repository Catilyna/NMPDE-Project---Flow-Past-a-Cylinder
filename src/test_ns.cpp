#include <iostream>
#include "NonStationaryNavierStokes.hpp"

int main(int argc, char* argv[])
{
    using namespace NavierStokes;
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    std::string mesh_file_name;
    // argument parsing for passing different mesh by terminal
    if(argc < 2){
        mesh_file_name = "../mesh/mesh3D_example.msh";

    } else{
        mesh_file_name = std::string(argv[1]); 
    }

    const unsigned int degree_velocity = 2;
    const unsigned int degree_pressure = 1;
    const double T = .5;              // final time (changed it for now just to save time)
    const double delta_t = 0.001;        // time step size
    const double theta = 1.0;          // parameter for the theta-method
    const double U_mean = 0.01;

    try {
            NonStationaryNavierStokes<3> flow(mesh_file_name, degree_velocity, degree_pressure, T, delta_t, theta, U_mean);
            flow.run_time_simulation();
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