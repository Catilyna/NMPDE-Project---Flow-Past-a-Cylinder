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
                             const dealii::TrilinosWrappers::BlockSparseMatrix &system_matrix,
                             const dealii::TrilinosWrappers::BlockSparseMatrix &pressure_mass,
                             const PreconditionerMp &Mppreconditioner);
    void vmult(BlockVector<double> &dst, const BlockVector<double> &src) const;
private:
    const double gamma;
    const double viscosity;
    const dealii::TrilinosWrappers::BlockSparseMatrix &system_matrix;
    const dealii::TrilinosWrappers::BlockSparseMatrix &pressure_mass;
    const PreconditionerMp &mp_preconditioner;
    SparseDirectUMFPACK A_inverse;
};
} // namespace NavierStokes
