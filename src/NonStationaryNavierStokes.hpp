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
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools_boundary.h>

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/solution_transfer.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <memory>

#include "preconditioners/BlockSchurPreconditioner.hpp"

using namespace dealii;

namespace NavierStokes{

    template<int dim>
    class NonStationaryNavierStokes {
        public:

            /** @brief class that defines the Inlet Velocity object. So the value
             * of the velocity of the fluid at the inlet (boundary).
             * It must be initialized with (dim + 1) component considering
             * pressure values aswell which will be set to 0.0 .
             */
            class InletVelocity : public Function<dim>
            {
            public:
                InletVelocity(const double U_mean)
                    : Function<dim>(dim + 1), U_mean(U_mean) {};

                virtual void
                vector_value(const Point<dim> &p, Vector<double> &values) const override
                {
                    double current_time = this->get_time();

                    double ramp_duration = 0.25;
                    double time_factor = 0.0;
                    if (current_time < ramp_duration)
                    {
                        time_factor = std::sin((M_PI / 8.0) * (current_time / ramp_duration));
                    }
                    else
                    {
                        time_factor = 1.0;
                    }
                    if(dim == 2){
                        values[0] = 4 * U_mean * p[1] * (H - p[1]) * time_factor / std::pow(H, 2.);
                    }
                    else if(dim == 3){
                        values[0] = 16 * U_mean * p[1] * p[2] * (H - p[1]) * (H - p[2]) * time_factor / std::pow(H, 4.);
                    }

                    for (unsigned int i = 1; i < dim + 1; ++i)
                        values[i] = 0.0;
                }

                virtual double
                value(const Point<dim> &p, const unsigned int component = 0) const override
                {
                    // 1. Copy the time logic exactly as it is in vector_value
                    double current_time = this->get_time();
                    double ramp_duration = 0.5;

                    double time_factor = 0.0;
                    if (current_time < ramp_duration)
                    {
                        time_factor = std::sin((M_PI / 2.0) * (current_time / ramp_duration));
                    }
                    else
                    {
                        time_factor = 1.0;
                    }

                    // 2. Return the value for the requested component
                    if (component == 0)
                    {
                        if(dim == 2){
                            return 4 * U_mean * p[1] * (H - p[1]) * time_factor / std::pow(H, 2.);
                        }
                        else if(dim == 3){
                            return 16 * U_mean * p[1] * p[2] * (H - p[1]) * (H - p[2]) * time_factor / std::pow(H, 4.);
                    }
                    }
                    else
                    {
                        // All other components (y-velocity, z-velocity, pressure) are zero
                        return 0.0;
                    }
                }

            protected:
                const double U_mean;
                const double H = 0.41; // Height is 0.41 in both 2D and 3D
                const double alpha = 1.0;
            };

            class InletVelocityTime : public Function<dim>
            {
            public:
                InletVelocityTime(const double U_mean)
                    : Function<dim>(dim + 1), U_mean(U_mean) {};

                virtual void
                vector_value(const Point<dim> &p, Vector<double> &values) const override
                {
                    double current_time = this->get_time();

                    double time_factor{ std::sin((M_PI / 8.0) * (current_time)) };
                   
                    if(dim == 2){
                        values[0] = 4 * U_mean * p[1] * (H - p[1]) * time_factor / std::pow(H, 2.);
                    } else if(dim == 3){
                        values[0] = 16 * U_mean * p[1] * p[2] * (H - p[1]) * (H - p[2]) * time_factor / std::pow(H, 4.);
                    }

                    for (unsigned int i = 1; i < dim + 1; ++i)
                        values[i] = 0.0;
                }

                virtual double
                value(const Point<dim> &p, const unsigned int component = 0) const override
                {
                    double current_time = this->get_time();

                    double time_factor{ std::sin((M_PI / 8.0) * (current_time)) };

                    if (component == 0)
                    {
                        if(dim == 2){
                            return 4 * U_mean * p[1] * (H - p[1]) * time_factor / std::pow(H, 2.);
                        } else if(dim == 3){
                            return 16 * U_mean * p[1] * p[2] * (H - p[1]) * (H - p[2]) * time_factor / std::pow(H, 4.);
                        }
                    }
                    else
                    {
                        // All other components (y-velocity, z-velocity, pressure) are zero
                        return 0.0;
                    }
                }

            protected:
                const double U_mean;
                const double H = 0.41; // Height is 0.41 in both 2D and 3D
                const double alpha = 1.0;
            };

            NonStationaryNavierStokes(const std::string &mesh_file_name_,
                                      const unsigned int &degree_velocity_,
                                      const unsigned int &degree_pressure_,
                                      const double &T_,
                                      const double &delta_t_,
                                      const double &theta_,
                                      const double &U_mean_,
                                      const double &viscosity_,
                                      const bool time_dependency_)
                : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)), mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)), pcout(std::cout, mpi_rank == 0), mesh_file_name(mesh_file_name_), degree_velocity(degree_velocity_), degree_pressure(degree_pressure_), T(T_), delta_t(delta_t_), theta(theta_), gamma(delta_t) // why??
                  , viscosity(viscosity_), U_mean(U_mean_), time_dependency(time_dependency_), mesh(MPI_COMM_WORLD) {};

            void run_time_simulation();

        private:
            void setup_dofs();

            void setup_boundaries();

            void initialize_system();

            void assemble(const bool initial_step, const bool assemble_matrix);

            void assemble_system(const bool initial_step);

            void assemble_rhs(const bool initial_step);

            void solve(const bool initial_step);

            void process_solution();

            void output_results() const;

            void newton_iteration(const double tolerance,
                                  const unsigned int max_n_line_searches,
                                  const bool is_initial_step,
                                  const bool output_result);

            void compute_initial_guess(double step_size);

            void set_initial_condition();

            void run();

            void compute_lift_drag();

            // problem related values setup
            const double viscosity;

            const double p_out = 1.;

            const double gamma;

            const unsigned int degree_velocity;
            const unsigned int degree_pressure;

            // parallelization setup
            const unsigned int mpi_size;
            const unsigned int mpi_rank;
            ConditionalOStream pcout;

            const std::string mesh_file_name;
            const bool time_dependency;
            const double U_mean;

            std::unique_ptr<Function<dim>> inlet_velocity_function;

            // InitialCondition initial_condition; ARE WE USING THIS??

            std::vector<types::global_dof_index> dofs_per_block;

            // Luca: I am quite confused about the template parameters here
            // G++ says that both are required, but Bucelli in his code only
            // specifies one (and it works). Need to clarify this point.
            parallel::fullydistributed::Triangulation<dim, dim> mesh;

            std::unique_ptr<FiniteElement<dim>> fe;

            std::unique_ptr<Quadrature<dim>> quadrature;
            std::unique_ptr<Quadrature<dim - 1>> quadrature_face;

            DoFHandler<dim> dof_handler;
            IndexSet locally_owned_dofs;
            IndexSet locally_relevant_dofs;

            std::vector<IndexSet> block_owned_dofs;
            std::vector<IndexSet> block_relevant_dofs;

            AffineConstraints<double> zero_constraints;

            AffineConstraints<double> nonzero_constraints;

            TrilinosWrappers::BlockSparsityPattern sparsity_pattern;

            TrilinosWrappers::BlockSparseMatrix system_matrix;
            TrilinosWrappers::BlockSparseMatrix pressure_mass;

            // following Bucelli's convention
            TrilinosWrappers::MPI::BlockVector solution_owned;
            TrilinosWrappers::MPI::BlockVector solution;
            TrilinosWrappers::MPI::BlockVector system_rhs;

            // Useful vectors used in Newton update
            TrilinosWrappers::MPI::BlockVector present_solution;
            TrilinosWrappers::MPI::BlockVector newton_update;
            TrilinosWrappers::MPI::BlockVector evaluation_point;

            // time stepping values
            const double T;               // final time
            const double delta_t;         // time step size
            double time = 0.0;            // current time
            unsigned int timestep_number; // current time step number

            const double theta; // parameter for the theta-method

            // old and current solution storage (for time stepping)
            TrilinosWrappers::MPI::BlockVector old_solution;
            TrilinosWrappers::MPI::BlockVector current_solution;
    };
}; // namespace NavierStokes
