#include "NavierStokesFractional.hpp"
#include <iostream>
#include <clipp.h>

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
    bool help = false;

    // define the dictionary with CLIPP
    using namespace clipp;
    auto cli = (
        (option("-f") & value("filename", mesh_file_name))  % "Path to the mesh file",
        (option("-v") & value("visc", viscosity))           % "Viscosity value (float)",
        (option("-theta") & value("theta", theta))          % "Theta parameter (float)",
        (option("-u") & value("max_u", U_max))              % "Max velocity (float)",
        (option("-d") & value("dim", dim))                  % "Dimension (2 or 3)",
        (option("-T") & value("T", T))                      % "Total simulation time (float)",
        (option("-dt") & value("delta_t", delta_t))         % "Time step size (float)",
        option("-h", "--help").set(help)                    % "Show this help message"
    );

    // Execute and manage the parsing
    if (!parse(argc, argv, cli)) {
        pcout << "Error on using comands.\n";
        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
            std::cout << make_man_page(cli, argv[0]);
        }
        return 1;
    }

    // Help (automatic) management 
    if (help) {
        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
            std::cout << make_man_page(cli, argv[0]);
        }
        return 0;
    }

    pcout << "Running with:" << std::endl;
    pcout << "  Mesh: " << mesh_file_name << std::endl;
    pcout << "  Viscosity: " << viscosity << std::endl;
    pcout << "  Theta: " << theta << std::endl;
    pcout << "  U_mean: " << U_max << std::endl;

    const unsigned int degree_velocity = 2;
    const unsigned int degree_pressure = 1;
    const double T = 2.;              
    const double delta_t = 0.01;       // time step size
    const bool time_dependency = true;
    try
    {
        if (dim == 2)
        {
            NavierStokesFractional<2> flow(mesh_file_name, 
                                              degree_velocity, 
                                              degree_pressure, 
                                              T, 
                                              delta_t,
                                              theta, 
                                              U_max, 
                                              viscosity, 
                                              time_dependency);
            flow.run_time_simulation();
        }
        else if (dim == 3)
        {
            NavierStokesFractional<3> flow(mesh_file_name, 
                                              degree_velocity, 
                                              degree_pressure, 
                                              T, 
                                              delta_t, 
                                              theta, 
                                              U_max, 
                                              viscosity, 
                                              time_dependency);
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