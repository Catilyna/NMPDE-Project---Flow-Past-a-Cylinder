#include "NonStationaryNavierStokes.hpp"

template <int dim>
class NavierStokesFractional : public NonStationaryNavierStokes<dim>
{
public:
    NavierStokesFractional(const std::string &mesh_file_name_,
                               const unsigned int &degree_velocity_,
                               const unsigned int &degree_pressure_,
                               const double &T_,
                               const double &delta_t_,
                               const double &theta_,
                               const double &U_mean_,
                               const double &viscosity_,
                               const bool time_dependency_)
        : NonStationaryNavierStokes<dim>(mesh_file_name_,
                                         degree_velocity_,
                                         degree_pressure_,
                                         T_,
                                         delta_t_,
                                         theta_,
                                         U_mean_,
                                         viscosity_,
                                         time_dependency_) {};

    void run_time_simulation() override;

private:
    // It changes every time step.
    TrilinosWrappers::BlockSparseMatrix step1_matrix;
    
    TrilinosWrappers::MPI::BlockVector step1_rhs;

    // only velocity is used.
    TrilinosWrappers::MPI::BlockVector solution_tilde;

    // This matrix is CONSTANT in time.
    TrilinosWrappers::BlockSparseMatrix step2_matrix;
    
    // RHS for Step 2.
    TrilinosWrappers::MPI::BlockVector step2_rhs;

    void setup_fractional_step_system();

    void assemble_step1_system();
    
    void assemble_step2_rhs();

    void solve_step1();
    
    void solve_step2();
};