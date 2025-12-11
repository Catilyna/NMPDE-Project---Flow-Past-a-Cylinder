#pragma once
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>
#include <fstream>
#include <iostream>
#include <vector>

namespace NavierStokes

using namespace dealii;

template <int dim>
class StationaryNavierStokes {
    public:
    StationaryNavierStokes(const unsigned int degree);
    void run(const unsigned int refinement);
private:
    void setup_dofs();

    void initialize_system();

    void assemble(const bool initial_step, const bool assemble_matrix);

    void assemble_system(const bool initial_step);

    void assemble_rhs(const bool initial_step);

    void solve(const bool initial_step);

    void refine_mesh();

    void process_solution(unsigned int refinement);

    void output_results(const unsigned int refinement_cycle) const;

    void newton_iteration(const double tolerance,
                         const unsigned int max_n_line_searches,
                         const unsigned int max_n_refinements,
                         const bool is_initial_step,
                         const bool output_result);

    void compute_initial_guess(double step_size);

    double viscosity;
    double gamma;
    const unsigned int degree;

    std::vector<types::global_dof_index> dofs_per_block;

    Triangulation<dim> triangulation;

    const FESystem<dim> fe;

    DoFHandler<dim> dof_handler;

    AffineConstraints<double> zero_constraints;

    AffineConstraints<double> nonzero_constraints;

    BlockSparsityPattern sparsity_pattern;

    BlockSparseMatrix<double> system_matrix;

    SparseMatrix<double> pressure_mass_matrix;

    BlockVector<double> present_solution;

    BlockVector<double> newton_update;

    BlockVector<double> system_rhs;

    BlockVector<double> evaluation_point;
}; // namespace NavierStokes
