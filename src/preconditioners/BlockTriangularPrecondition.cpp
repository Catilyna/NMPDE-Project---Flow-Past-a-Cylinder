#include "BlockTriangularPrecondition.hpp"

#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_control.h>

namespace NavierStokes {

    void
    PreconditionBlockTriangular::initialize(const TrilinosWrappers::SparseMatrix &velocity_stiffness_,
                                            const TrilinosWrappers::SparseMatrix &pressure_mass_,
                                            const TrilinosWrappers::SparseMatrix &B_)
    {
        velocity_stiffness = &velocity_stiffness_;
        pressure_mass      = &pressure_mass_;
        B                  = &B_;

        preconditioner_velocity.initialize(velocity_stiffness_);
        preconditioner_pressure.initialize(pressure_mass_);
    }

    void
    PreconditionBlockTriangular::vmult(TrilinosWrappers::MPI::BlockVector       &dst,
                                       const TrilinosWrappers::MPI::BlockVector &src) const
    {
        SolverControl solver_control_velocity(20000, 1e-2 * src.block(0).l2_norm());
        
        SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres_velocity(solver_control_velocity);
        solver_gmres_velocity.solve(*velocity_stiffness,
                       dst.block(0),
                       src.block(0),
                       preconditioner_velocity);
        
        tmp.reinit(src.block(1));
        
        B->vmult(tmp, dst.block(0));
        
        tmp.sadd(-1.0, src.block(1));

        SolverControl solver_control_pressure(20000, 1e-2 * src.block(1).l2_norm());
            
        SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres_pressure(solver_control_pressure);
        solver_gmres_pressure.solve(*pressure_mass,
                       dst.block(1),
                       tmp,
                       preconditioner_pressure);
    }

}