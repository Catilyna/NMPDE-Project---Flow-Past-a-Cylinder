#pragma once
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>

namespace NavierStokes {

class PressureConvectionDiffusionPCD {
public:
	PressureConvectionDiffusionPCD() = default; // default constructor

    // Initialize the preconditioner with the necessary matrices
	void initialize(const dealii::TrilinosWrappers::SparseMatrix &pressure_mass_,
				   const dealii::TrilinosWrappers::SparseMatrix &pressure_laplace_,
				   const dealii::TrilinosWrappers::SparseMatrix &pressure_conv_diff_);
    
    // Apply the preconditioner to a vector
	void vmult(dealii::TrilinosWrappers::MPI::Vector &dst,
			   const dealii::TrilinosWrappers::MPI::Vector &src) const;

private:
	const dealii::TrilinosWrappers::SparseMatrix *pressure_mass = nullptr;       // pointer to pressure mass matrix (pointer to avoid copying, no smart pointer since we do not deal with ownership here)
	const dealii::TrilinosWrappers::SparseMatrix *pressure_laplace = nullptr;    // pointer to pressure laplace matrix
	const dealii::TrilinosWrappers::SparseMatrix *pressure_conv_diff = nullptr;  // pointer to pressure convection-diffusion matrix

    // ILU preconditioners for the individual blocks
	mutable dealii::TrilinosWrappers::PreconditionILU ilu_mass;
	mutable dealii::TrilinosWrappers::PreconditionILU ilu_laplace;
	mutable dealii::TrilinosWrappers::PreconditionILU ilu_conv_diff;
};

} // namespace NavierStokes
