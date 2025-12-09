#ifndef PRECONDITION_BLOCK_TRIANGULAR_HPP
#define PRECONDITION_BLOCK_TRIANGULAR_HPP

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>

namespace NavierStokes {
    using namespace dealii;

    /**
     * @brief A Block Triangular Preconditioner.
     * * This class implements a Lower Block Triangular preconditioner.
     * Unlike the Block Diagonal version, this accounts for the coupling 
     * from Velocity to Pressure via the B matrix (Divergence).
     * * The application steps are:
     * 1. Solve A * u = f (Velocity)
     * 2. Calculate residual transfer: r_p_new = B * u - g
     * 3. Solve Mp * p = r_p_new (Pressure)
     */
    class PreconditionBlockTriangular
    {
    public:
        /**
         * @brief Initialize the preconditioner.
         * * Sets up pointers to the system matrices and initializes the inner 
         * ILU preconditioners.
         */
        void
        initialize(const TrilinosWrappers::SparseMatrix &velocity_stiffness_,
                   const TrilinosWrappers::SparseMatrix &pressure_mass_,
                   const TrilinosWrappers::SparseMatrix &B_);

        /**
         * @brief Application of the preconditioner.
         * * Performs the block triangular solve logic.
         */
        void
        vmult(TrilinosWrappers::MPI::BlockVector       &dst,
              const TrilinosWrappers::MPI::BlockVector &src) const;

    protected:
        const TrilinosWrappers::SparseMatrix *velocity_stiffness;
        const TrilinosWrappers::SparseMatrix *pressure_mass;
        const TrilinosWrappers::SparseMatrix *B;

        TrilinosWrappers::PreconditionILU preconditioner_velocity;
        TrilinosWrappers::PreconditionILU preconditioner_pressure;

        // Temporary vector for intermediate calculations (B * u)
        // Marked mutable because vmult is const but modifies this internal workspace.
        mutable TrilinosWrappers::MPI::Vector tmp;
    };

}

#endif 