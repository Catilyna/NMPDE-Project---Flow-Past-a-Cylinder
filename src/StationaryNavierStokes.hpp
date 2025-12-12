#pragma once
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

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
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>
#include <fstream>
#include <iostream>
#include <vector>

using namespace dealii;

namespace NavierStokes{

    template<int dim>
    class StationaryNavierStokes {
        public:

        class InletVelocity : public Function<dim>
        {
        public:
            InletVelocity()
            : Function<dim>(dim + 1)
            {}

            virtual void
            vector_value(const Point<dim> &p, Vector<double> &values) const override
            {
            values[0] = 1.0;

            for (unsigned int i = 1; i < dim + 1; ++i)
                values[i] = 0.0;
            }

            virtual double
            value(const Point<dim> &p, const unsigned int component = 0) const override
            {
            if (component == 0)
                return 1.0;
            else
                return 0.0;
            }

        protected:
            const double alpha = 1.0;
        };

        StationaryNavierStokes(const std::string  &mesh_file_name_,)
            : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
            , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
            , pcout(std::cout, mpi_rank == 0)
            , mesh_file_name(mesh_file_name_)
            , mesh(MPI_COMM_WORLD)
        {};
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

        const unsigned int mpi_size;

        const unsigned int mpi_rank;

        const std::string mesh_file_name;

        ConditionalOStream pcout;

        InletVelocity inlet_velocity;

        std::vector<types::global_dof_index> dofs_per_block;

        parallel::fullydistributed::Triangulation<dim> triangulation;

        std::unique_ptr<FiniteElement<dim>> fe;

        std::unique_ptr<Quadrature<dim>> quadrature;
        std::unique_ptr<Quadrature<dim - 1>> quadrature_face;

        DoFHandler<dim> dof_handler;
        IndexSet locally_owned_dofs;
        std::vector<IndexSet> block_owned_dofs;

        AffineConstraints<double> zero_constraints;

        AffineConstraints<double> nonzero_constraints;

        TrilinosWrappers::BlockSparsityPattern sparsity_pattern;

        TrilinosWrappers::BlockSparseMatrix system_matrix;
        TrilinosWrappers::BlockSparseMatrix pressure_mass;

        // following Bucelli's convention
        TrilinosWrappers::MPI::BlockVector solution_owned;
        TrilinosWrappers::MPI::BlockVector solution;

        TrilinosWrappers::MPI::BlockVector system_rhs;
        BlockVector<double> newton_update;

        BlockVector<double> evaluation_point;
    };
}; // namespace NavierStokes
