#include "PreconditionIdentity.hpp"

namespace NavierStokes {
    void 
    PreconditionIdentity::vmult(TrilinosWrappers::MPI::BlockVector       &dst,
                                const TrilinosWrappers::MPI::BlockVector &src) const
    {
        // The Identity operation: dst = I * src
        // Which simplifies to a direct copy.
        dst = src;
    }

}