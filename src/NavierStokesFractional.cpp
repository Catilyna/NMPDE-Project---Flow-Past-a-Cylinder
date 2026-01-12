#include "NavierStokesFractional.hpp"
#include <iomanip>
#include "NonStationaryNavierStokes.hpp"
#include "./preconditioners/BlockPrecondtioner.h"
#include "./preconditioners/BlockSchurPreconditioner.hpp"
#include "./preconditioners/BlockTriangularPrecondition.hpp"
#include "./preconditioners/PreconditionIdentity.h"
// Helper for .pvd file management
#include <cstdio>
#include <sstream>

template <int dim>
void NavierStokesFractional<dim>::setup_fractional_step_system()
{
    this->initialize_system();
    this->set_initial_condition();

    step1_matrix.reinit(this->sparsity_pattern);
    step2_matrix.reinit(this->sparsity_pattern);

    // initialize Vectors
    solution_tilde.reinit(this->block_owned_dofs, MPI_COMM_WORLD);
    step1_rhs.reinit(this->block_owned_dofs, MPI_COMM_WORLD);
    step2_rhs.reinit(this->block_owned_dofs, MPI_COMM_WORLD);

    // We assume you create a helper function similar to your standard assemble.
    this->pcout << "Assembling constant projection matrix..." << std::endl;
    assemble_constant_step2_matrix(step2_matrix); 
}

template <int dim>
void NavierStokesFractional<dim>::solve_step1()
{
    SolverControl solver_control(10000, 1e-12, true);
    TrilinosWrappers::SolverGMRES solver(solver_control);

    // Preconditioner: AMG (ML or MueLu) is perfect for Advection-Diffusion
    TrilinosWrappers::PreconditionAMG preconditioner;
    TrilinosWrappers::PreconditionAMG::AdditionalData data;
    
    preconditioner.initialize(step1_matrix.block(0,0), data);

    // Solve strictly on Block 0
    solver.solve(step1_matrix.block(0,0), 
                 solution_tilde.block(0), 
                 step1_rhs.block(0), 
                 preconditioner);

    // set pressure DoFs to 0
    solution_tilde.block(1) = 0;
}

template <int dim>
void NavierStokesFractional<dim>::solve_step2()
{
    SolverControl solver_control(10000, 1e-12, true);
    TrilinosWrappers::SolverGMRES solver(solver_control);

    // Preconditioner: AMG (ML or MueLu) is perfect for Advection-Diffusion
    TrilinosWrappers::PreconditionAMG preconditioner;
    TrilinosWrappers::PreconditionAMG::AdditionalData data;
    
    preconditioner.initialize(step2_matrix, data);

    // Solve strictly on Block 0
    solver.solve(step2_matrix, 
                 this->solution, 
                 step2_rhs, 
                 preconditioner);

    // set pressure DoFs to 0
    solution_tilde.block(1) = 0;
}

template<int dim>
void NavierStokesFractional<dim>::assemble_step1_system()
{

}

template<int dim>
void NavierStokesFractional<dim>::assemble_step2_system()
{
    
}

template <int dim>
void NavierStokesFractional<dim>::run_time_simulation()
{
    setup_fractional_step_system();

    double time = 0.0;
    double time_step = this->time_step; // From base class

    while (time < this->end_time)
    {
        time += time_step;
        this->pcout << "Time step: " << time << std::endl;

        // --- STEP 1: Predict Velocity (tilde_u) ---
        // Uses u^n (current solution) to build matrix/RHS
        assemble_step1_system(); 
        solve_step1(); // Result stored in 'solution_tilde'
        
        // Apply Dirichlet BCs to intermediate velocity if needed
        this->constraints.distribute(solution_tilde);

        // --- STEP 2: Project (u^{n+1}, p^{n+1}) ---
        // Uses 'solution_tilde' to build the RHS
        assemble_step2_rhs(); 
        solve_step2(); // Result stored in 'this->solution' (u^n+1)

        // Update for next step
        // (In this scheme, 'this->solution' is now ready for the next step)
        
        this->output_results(time);
    }
}