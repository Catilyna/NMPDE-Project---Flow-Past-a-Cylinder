#include "NonStationaryNavierStokes.hpp"
#include "Timer.hpp"
#include <iostream>

int main(int argc, char* argv[])
{
    using namespace NavierStokes;
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    
    std::vector<std::string> args(argv + 1, argv + argc);

    // set some defaults
    std::string mesh_file_name = "../mesh/mesh3D_example.msh";
    double viscosity = 1.;
    double theta = 1.; // parameter for the theta method
    double U_max = 0.45;
    int dim = 3;

    for (size_t i = 0; i < args.size();++i){
        if (args[i] == "-h" || args[i] == "--help") {
            pcout << "Usage: ...\n";
            return 0;
        }
        else if(args[i] == "-f"){
            if (i+1 < args.size()) mesh_file_name = args[++i]; // increment i and assign the name to the mesh_filename
            else {
                pcout << "-f requires an argument..." << std::endl;
                pcout << "Exiting..." << std::endl;
                return 1;
            }
        }
        else if(args[i] == "-v"){
            if(i+1 < args.size())
                viscosity = std::stod(args[++i]); // increment i and assign to viscosity value
            else{
                pcout << "-v requires a float argument..." << std::endl;
                pcout << "Exiting..." << std::endl;
                return 1;
            }
        }
        else if(args[i] == "-theta"){
            if(i+1 < args.size())
                theta = std::stod(args[++i]);
            else{
                pcout << "-theta requires a float argument..." << std::endl;
                pcout << "Exiting..." << std::endl;
                return 1;
            }
        }
        else if(args[i] == "-u"){
            if(i+1 < args.size()){
                U_max = std::stod(args[++i]);
            }
            else{
                pcout << "-u requires a float argument..." << std::endl;
                pcout << "Exiting..." << std::endl;
                return 1;
            }
        }
        else if(args[i] == "-d") {
            if(i + 1 < args.size()) {
                dim = std::stoi(args[++i]);
            }else {
                pcout << "-d requires a interger argument..." << std::endl;
                pcout << "Exiting..." << std::endl;
                return 1;
            }
        }
    }

    pcout << "Running with:" << std::endl;
    pcout << "  Mesh: " << mesh_file_name << std::endl;
    pcout << "  Viscosity: " << viscosity << std::endl;
    pcout << "  Theta: " << theta << std::endl;
    pcout << "  U_max: " << U_max << std::endl;

    const unsigned int degree_velocity = 2;
    const unsigned int degree_pressure = 1;
    const double T = 4.;              
    const double delta_t = 0.02;       // time step size
    const bool time_dependency = false;
    try
    {
        if (dim == 2)
        {
            NonStationaryNavierStokes<2> flow(
                mesh_file_name,
                degree_velocity,
                degree_pressure,
                T,
                delta_t,
                theta,
                U_max,
                viscosity,
                time_dependency
            );
            flow.run_time_simulation();
        }
        else if (dim == 3)
        {
            NonStationaryNavierStokes<3> flow(
                mesh_file_name,
                degree_velocity,
                degree_pressure,
                T,
                delta_t,
                theta,
                U_max,
                viscosity,
                time_dependency
            );
            flow.run_time_simulation();
        }
        else
        {
            pcout << "Error: Dimension must be 2 or 3." << std::endl;
            return 1;
        }
        return 0;
    }
    catch (std::exception &exc)
    {
        pcout << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl
              << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
        return 1;
    }
    catch (...)
    {
        pcout << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl
              << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
        return 1;
    }
}