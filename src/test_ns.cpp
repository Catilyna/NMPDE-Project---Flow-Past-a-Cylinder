#include "NonStationaryNavierStokes.hpp"
#include "Timer.hpp"
#include <iostream>

int main(int argc, char* argv[])
{
    using namespace NavierStokes;
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    std::vector<std::string> args(argv + 1, argv + argc);

    // set some defaults
    std::string mesh_file_name = "../mesh/mesh3D_example.msh";
    double viscosity = 1.;
    double theta = 1.; // parameter for the theta method
    double U_mean = 0.45;
    int dim = 3;

    for (int i = 0; i < args.size();++i){
        if (args[i] == "-h" || args[i] == "--help") {
            std::cout << "Usage: ...\n";
            return 0;
        }
        else if(args[i] == "-f"){
            if (i+1 < args.size()) mesh_file_name = args[++i]; // increment i and assign the name to the mesh_filename
            else {
                std::cout << "-f requires an argument..." << std::endl;
                std::cout << "Exiting..." << std::endl;
                return 1;
            }
        }
        else if(args[i] == "-v"){
            if(i+1 < args.size())
                viscosity = std::stod(args[++i]); // increment i and assign to viscosity value
            else{
                std::cout << "-v requires a float argument..." << std::endl;
                std::cout << "Exiting..." << std::endl;
                return 1;
            }
        }
        else if(args[i] == "-theta"){
            if(i+1 < args.size())
                theta = std::stod(args[++i]);
            else{
                std::cout << "-theta requires a float argument..." << std::endl;
                std::cout << "Exiting..." << std::endl;
                return 1;
            }
        }
        else if(args[i] == "-u"){
            if(i+1 < args.size()){
                U_mean = std::stod(args[++i]);
            }
            else{
                std::cout << "-u requires a float argument..." << std::endl;
                std::cout << "Exiting..." << std::endl;
                return 1;
            }
        }
        else if(args[i] == "-d") {
            if(i + 1 < args.size()) {
                dim = std::stod(args[++i]);
            }else {
                std::cout << "-d requires a interger argument..." << std::endl;
                std::cout << "Exiting..." << std::endl;
                return 1;
            }
        }
    }

    std::cout << "Running with:" << std::endl;
    std::cout << "  Mesh: " << mesh_file_name << std::endl;
    std::cout << "  Viscosity: " << viscosity << std::endl;
    std::cout << "  Theta: " << theta << std::endl;
    std::cout << "  U_mean: " << U_mean << std::endl;

    const unsigned int degree_velocity = 2;
    const unsigned int degree_pressure = 1;
    const double T = 5.;              
    const double delta_t = 0.02;       // time step size
    const bool time_dependency = true;
    try
    {
        if (dim == 2)
        {
            NonStationaryNavierStokes<2> flow(mesh_file_name, 
                                              degree_velocity, 
                                              degree_pressure, 
                                              T, 
                                              delta_t,
                                              theta, 
                                              U_mean, 
                                              viscosity, 
                                              time_dependency);
            ScopedTimer("Navier Stokes Simulation");
            flow.run_time_simulation();
        }
        else if (dim == 3)
        {
            NonStationaryNavierStokes<3> flow(mesh_file_name, 
                                              degree_velocity, 
                                              degree_pressure, 
                                              T, 
                                              delta_t, 
                                              theta, 
                                              U_mean, 
                                              viscosity, 
                                              time_dependency);
            ScopedTimer("Navier Stokes Simulation");
            flow.run_time_simulation();
        }
        else
        {
            std::cerr << "Error: Dimension must be 2 or 3." << std::endl;
            return 1;
        }
        return 0;
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------" << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------" << std::endl;
        return 1;
    }
}