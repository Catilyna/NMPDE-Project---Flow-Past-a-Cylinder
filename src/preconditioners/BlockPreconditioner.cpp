#include "BlockPreconditioner.hpp"

// Headers required for the implementation of vmult
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

namespace NavierStokes {

    void
    BlockPreconditioner::initialize(const TrilinosWrappers::SparseMatrix &velocity_stiffness_,
                                          const TrilinosWrappers::SparseMatrix &pressure_mass_)
    {
        // Store pointers to the matrices
        velocity_stiffness = &velocity_stiffness_;
        pressure_mass      = &pressure_mass_;

        // Initialize the specific ILU preconditioners for the inner solves.
        // This calculates the incomplete factorization L and U.
        preconditioner_velocity.initialize(velocity_stiffness_);
        preconditioner_pressure.initialize(pressure_mass_);
    }

    void
    BlockPreconditioner::vmult(TrilinosWrappers::MPI::BlockVector       &dst,
                                     const TrilinosWrappers::MPI::BlockVector &src) const
    {
        // --- 1. Solve the Velocity Block (A * u = r_u) ---
        // We set a relative tolerance based on the residual norm.
        SolverControl solver_control_velocity(1000, 1e-2 * src.block(0).l2_norm());
        
        SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_velocity(solver_control_velocity);

        solver_cg_velocity.solve(*velocity_stiffness,
                                 dst.block(0),
                                 src.block(0),
                                 preconditioner_velocity);

        SolverControl solver_control_pressure(1000, 1e-2 * src.block(1).l2_norm());
        
        SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_pressure(solver_control_pressure);

        solver_cg_pressure.solve(*pressure_mass,
                                 dst.block(1),
                                 src.block(1),
                                 preconditioner_pressure);
    }

}