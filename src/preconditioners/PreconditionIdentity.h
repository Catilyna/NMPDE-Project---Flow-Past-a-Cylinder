#ifndef PRECONDITION_IDENTITY_HPP
#define PRECONDITION_IDENTITY_HPP

#include <deal.II/lac/trilinos_block_vector.h>

namespace NavierStokes {
    using namespace dealii;

    /**
     * @brief A simple identity preconditioner.
     * * This class acts as a placeholder preconditioner that performs
     * no modification to the input vector. It effectively applies
     * the Identity matrix (I).
     */
    class PreconditionIdentity
    {
    public:
        /**
         * @brief Application of the preconditioner.
         * * Copies the source vector (src) directly into the destination 
         * vector (dst) without modification.
         */
        void vmult(TrilinosWrappers::MPI::BlockVector       &dst,
                   const TrilinosWrappers::MPI::BlockVector &src) const;

    protected:
    };

}

#endif