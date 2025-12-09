#ifndef PRECONDITION_BLOCK_DIAGONAL_HPP
#define PRECONDITION_BLOCK_DIAGONAL_HPP

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>

namespace NavierStokes{
    using namespace dealii;

    class BlockPrecondtioner{
        public:
            /**
             * @brief Initialize the preconditioner.
             * * Sets up the internal ILU preconditioners for both the velocity
             * and pressure blocks.
             */
            void initialize(const TrilinosWrappers::SparseMatrix &velocity_stiffness_,
                            const TrilinosWrappers::SparseMatrix &pressure_mass_);

            /**
             * @brief Application of the preconditioner.
             * * Solves A * u_dst = u_src and Mp * p_dst = p_src
             * using the Conjugate Gradient method.
             */
            void vmult(TrilinosWrappers::MPI::BlockVector        &dst,
                        const TrilinosWrappers::MPI::BlockVecotr &src) const;

        protected:
            const TrilinosWrappers::SparseMatrix *velocity_stiffness;
            const TrilinosWrappers::SparseMatrix *pressure_mass;

            TrilinosWrappers::PreconditionILU preconditioner_velocity;
            TrilinosWrappers::PreconditionILU preconditioner_pressure;
    }
}

#endif
