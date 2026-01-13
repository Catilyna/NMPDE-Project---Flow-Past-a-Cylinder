#pragma once
#include <deal.II/base/subscriptor.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>   
#include <deal.II/lac/trilinos_vector.h>        
#include <deal.II/lac/trilinos_sparse_matrix.h>  
#include <deal.II/lac/trilinos_block_sparse_matrix.h> 
#include <deal.II/lac/trilinos_precondition.h>   
#include <deal.II/lac/solver_cg.h>    
#include <deal.II/lac/solver_gmres.h>           
#include <deal.II/lac/solver_control.h>        
#include <deal.II/lac/sparse_direct.h>  

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
    void vmult(TrilinosWrappers::MPI::BlockVector &dst, const TrilinosWrappers::MPI::BlockVector &src) const;
private:
    const double gamma;
    const double viscosity;
    const dealii::TrilinosWrappers::BlockSparseMatrix &system_matrix;
    const dealii::TrilinosWrappers::BlockSparseMatrix &pressure_mass;
    const PreconditionerMp &mp_preconditioner;
    TrilinosWrappers::PreconditionILU A_inverse;
};
} // namespace NavierStokes
