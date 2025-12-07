#include "BlockSchurPreconditioner.h"
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>

namespace NavierStokes {
using namespace dealii;

template <class PreconditionerMp>
BlockSchurPreconditioner<PreconditionerMp>::BlockSchurPreconditioner(
    double gamma,
    double viscosity,
    const BlockSparseMatrix<double> &S,
    const SparseMatrix<double> &P,
    const PreconditionerMp &Mppreconditioner)
    : gamma(gamma), viscosity(viscosity), stokes_matrix(S), pressure_mass_matrix(P), mp_preconditioner(Mppreconditioner) {
    A_inverse.initialize(stokes_matrix.block(0, 0));
}

template <class PreconditionerMp>
void BlockSchurPreconditioner<PreconditionerMp>::vmult(
    BlockVector<double> &dst, const BlockVector<double> &src) const {
    Vector<double> utmp(src.block(0));
    {
        SolverControl solver_control(1000, 1e-6 * src.block(1).l2_norm());
        SolverCG<Vector<double>> cg(solver_control);
        dst.block(1) = 0.0;
        cg.solve(pressure_mass_matrix, dst.block(1), src.block(1), mp_preconditioner);
        dst.block(1) *= -(viscosity + gamma);
    }
    {
        stokes_matrix.block(0, 1).vmult(utmp, dst.block(1));
        utmp *= -1.0;
        utmp += src.block(0);
    }
    A_inverse.vmult(dst.block(0), utmp);
}

// Explicit instantiation for SparseILU<double>
template class BlockSchurPreconditioner<SparseILU<double>>;
} // namespace NavierStokes
