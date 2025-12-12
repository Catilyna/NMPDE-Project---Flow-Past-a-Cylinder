#pragma once
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/base/subscriptor.h>

namespace NavierStokes {
using namespace dealii;

template <class PreconditionerMp>
class BlockSchurPreconditioner : public Subscriptor {
public:
    BlockSchurPreconditioner(double gamma,
                             double viscosity,
                             const BlockSparseMatrix<double> &S,
                             const SparseMatrix<double> &P,
                             const PreconditionerMp &Mppreconditioner);
    void vmult(BlockVector<double> &dst, const BlockVector<double> &src) const;
private:
    const double gamma;
    const double viscosity;
    const BlockSparseMatrix<double> &stokes_matrix;
    const SparseMatrix<double> &pressure_mass_matrix;
    const PreconditionerMp &mp_preconditioner;
    SparseDirectUMFPACK A_inverse;
};
} // namespace NavierStokes
