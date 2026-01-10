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

    ilu_mass.initialize(pressure_mass_);
    ilu_laplace.initialize(pressure_laplace_);
    ilu_conv_diff.initialize(pressure_conv_diff_);
}

void PressureConvectionDiffusionPCD::vmult(
    dealii::TrilinosWrappers::MPI::Vector &dst,
    const dealii::TrilinosWrappers::MPI::Vector &src) const
{
    // Implements: dst = M_p^{-1} K_p A_p^{-1} M_p^{-1} src
    dealii::TrilinosWrappers::MPI::Vector tmp1, tmp2;
    tmp1.reinit(src);
    tmp2.reinit(src);

    // Step 1: tmp1 = M_p^{-1} src
    {
        SolverControl control(1000, 1e-8 * src.l2_norm());
        SolverGMRES<dealii::TrilinosWrappers::MPI::Vector> solver(control);
        solver.solve(*pressure_mass, tmp1, src, ilu_mass);
    }

    // Step 2: tmp2 = A_p^{-1} tmp1 (A_p = convection-diffusion)
    {
        SolverControl control(1000, 1e-8 * tmp1.l2_norm());
        SolverGMRES<dealii::TrilinosWrappers::MPI::Vector> solver(control);
        solver.solve(*pressure_conv_diff, tmp2, tmp1, ilu_conv_diff);
    }

    // Step 3: tmp1 = K_p tmp2 (K_p = pressure Laplacian)
    pressure_laplace->vmult(tmp1, tmp2);

    // Step 4: dst = M_p^{-1} tmp1
    {
        SolverControl control(1000, 1e-8 * tmp1.l2_norm());
        SolverGMRES<dealii::TrilinosWrappers::MPI::Vector> solver(control);
        solver.solve(*pressure_mass, dst, tmp1, ilu_mass);
    }
}

} // namespace NavierStokes
