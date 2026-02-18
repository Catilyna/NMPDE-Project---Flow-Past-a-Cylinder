#include "NonStationaryNavierStokes.hpp"
#include "Timer.hpp"
#include <iostream>
#include <clipp.h>

int main(int argc, char* argv[])
{
    using namespace NavierStokes;
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    
    std::vector<std::string> args(argv + 1, argv + argc);

    // set some defaults
    std::string mesh_file_name = "../mesh/mesh3D_coarse_cylinder.msh";
    double viscosity = 1.;
    double theta = 1.; // parameter for the theta method
    double U_max = 0.45;
    int dim = 3;
    bool help = false;

    constexpr int degree_velocity = 2;
    constexpr int degree_pressure = 1;
    double T = 2.;              
    double delta_t = 0.01;       // time step size
    bool t_dep = false;    // inlet velocity is sinusoidal (true) or a ramp (false)

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
        (option("-td") & value("t dep", t_dep))             % "Variable inlet velocity",
        option("-h", "--help").set(help)                    % "Show this help message"
    );

    // Execute and manage the parsing
    if (!parse(argc, argv, cli)) {
        pcout << "Error on using comands.\n";
        pcout << make_man_page(cli, argv[0]);
        return 1;
    }

    // Help (automatic) management 
    if (help) {
        pcout << make_man_page(cli, argv[0]);
        return 0;
    }

    pcout << "Running with:" << std::endl;
    pcout << "  Mesh: " << mesh_file_name << std::endl;
    pcout << "  Viscosity: " << viscosity << std::endl;
    pcout << "  Theta: " << theta << std::endl;
    pcout << "  U_max: " << U_max << std::endl;

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
                t_dep
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
                t_dep
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