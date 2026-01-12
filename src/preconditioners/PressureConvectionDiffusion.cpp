#include "PressureConvectionDiffusion.hpp"
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>

namespace NavierStokes {

void PressureConvectionDiffusionPCD::initialize(
    const dealii::TrilinosWrappers::SparseMatrix &pressure_mass_,
    const dealii::TrilinosWrappers::SparseMatrix &pressure_laplace_,
    const dealii::TrilinosWrappers::SparseMatrix &pressure_conv_diff_)
{
    pressure_mass = &pressure_mass_;
    pressure_laplace = &pressure_laplace_;
    pressure_conv_diff = &pressure_conv_diff_;

    dealii::TrilinosWrappers::PreconditionILU::AdditionalData data;
    data.ilu_fill = 1.0;
    ilu_mass.initialize(pressure_mass_, data);
    ilu_laplace.initialize(pressure_laplace_, data);
    ilu_conv_diff.initialize(pressure_conv_diff_, data);
}

void PressureConvectionDiffusionPCD::vmult(
    dealii::TrilinosWrappers::MPI::Vector &dst,
    const dealii::TrilinosWrappers::MPI::Vector &src) const
{
    // Implements: dst = M_p^{-1} K_p A_p^{-1} K_p M_p^{-1} src
    dealii::TrilinosWrappers::MPI::Vector tmp1, tmp2;
    tmp1.reinit(src);
    tmp2.reinit(src);

    // Step 1: tmp1 = M_p^{-1} src
    {
        double tol = std::max(1e-8 * src.l2_norm(), 1e-12);
        dealii::SolverControl control(1000, tol);
        dealii::SolverCG<dealii::TrilinosWrappers::MPI::Vector> solver(control);
        solver.solve(*pressure_mass, tmp1, src, ilu_mass);
    }

    // Step 2: tmp2 = K_p tmp1
    pressure_laplace->vmult(tmp2, tmp1);

    // Step 3: tmp1 = A_p^{-1} tmp2
    {
        double tol = std::max(1e-8 * tmp2.l2_norm(), 1e-12);
        dealii::SolverControl control(1000, tol);
        dealii::SolverCG<dealii::TrilinosWrappers::MPI::Vector> solver(control);
        solver.solve(*pressure_conv_diff, tmp1, tmp2, ilu_conv_diff);
    }

    // Step 4: tmp2 = K_p tmp1
    pressure_laplace->vmult(tmp2, tmp1);

    // Step 5: dst = M_p^{-1} tmp2
    {
        double tol = std::max(1e-8 * tmp2.l2_norm(), 1e-12);
        dealii::SolverControl control(1000, tol);
        dealii::SolverCG<dealii::TrilinosWrappers::MPI::Vector> solver(control);
        solver.solve(*pressure_mass, dst, tmp2, ilu_mass);
    }
}

} // namespace NavierStokes
