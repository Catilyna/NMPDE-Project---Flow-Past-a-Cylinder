#include "BlockSchurPreconditioner.hpp"

namespace NavierStokes {
    using namespace dealii;

    template <class PreconditionerMp>
    BlockSchurPreconditioner<PreconditionerMp>::BlockSchurPreconditioner(
        double gamma,
        double viscosity,
        const TrilinosWrappers::BlockSparseMatrix &S, // Changed to Trilinos
        const TrilinosWrappers::BlockSparseMatrix &P,
        const PreconditionerMp &Mppreconditioner)
        : gamma(gamma), viscosity(viscosity), system_matrix(S), pressure_mass(P), mp_preconditioner(Mppreconditioner) {
        A_inverse.initialize(system_matrix.block(0, 0));
    }

    // many changes where required here as we are not using Block Vectors anymore but Trilinos' version.
    template <class PreconditionerMp>
    void BlockSchurPreconditioner<PreconditionerMp>::vmult(
        TrilinosWrappers::MPI::BlockVector &dst, const TrilinosWrappers::MPI::BlockVector &src) const {
        TrilinosWrappers::MPI::Vector utmp;
        utmp.reinit(src.block(0));
        {
            const double target_tolerance = std::max(1e-2 * src.block(1).l2_norm(), 1e-20);

            // 2. Set SolverControl.
            // We allow fewer iterations (1000 is usually plenty for an inner solve).
            SolverControl solver_control(1000, target_tolerance);

            SolverFGMRES<TrilinosWrappers::MPI::Vector> cg(solver_control);

            dst.block(1) = 0.0;
            try {
                cg.solve(pressure_mass.block(1, 1), dst.block(1), src.block(1), mp_preconditioner);
            } catch (std::exception &e) {

                std::cerr << "Inner preconditioner warning: " << e.what() << std::endl;
            }

            dst.block(1) *= -(viscosity + gamma);
            /*
            SolverControl solver_control(10000, 1e-6 * src.block(1).l2_norm());
            SolverFGMRES<TrilinosWrappers::MPI::BlockVector> cg(solver_control);
            dst.block(1) = 0.0;
            cg.solve(pressure_mass.block(1, 1), dst.block(1), src.block(1), mp_preconditioner);
            dst.block(1) *= -(viscosity + gamma);*/
        }
        {
            system_matrix.block(0, 1).vmult(utmp, dst.block(1));
            utmp.sadd(-1.0, 1.0, src.block(0)); // use the mpi aware operation
        }
        A_inverse.vmult(dst.block(0), utmp);
    }

    // Explicit instantiation for SparseILU<double>
    template class BlockSchurPreconditioner<TrilinosWrappers::PreconditionILU>;
} // namespace NavierStokes
