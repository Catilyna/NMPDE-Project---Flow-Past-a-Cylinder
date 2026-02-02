#include "NavierStokesFractional.hpp"

#include <iomanip>
#include <cstdio>
#include <sstream>

template <int dim>
void NavierStokesFractional<dim>::setup_fractional_step_system()
{
    this->initialize_system(); // call this method from the base class to initialize base members
    this->set_initial_condition();
    solution = 0;

    step1_matrix.reinit(this->sparsity_pattern);
    step2_matrix.reinit(this->sparsity_pattern);
    step3_matrix.reinit(this->sparsity_pattern);

    // initialize Vectors
    solution_tilde.reinit(this->block_owned_dofs, MPI_COMM_WORLD);
    step1_rhs.reinit(this->block_owned_dofs, MPI_COMM_WORLD);
    step2_rhs.reinit(this->block_owned_dofs, MPI_COMM_WORLD);
    step3_rhs.reinit(this->block_owned_dofs, MPI_COMM_WORLD);
}

template<int dim>
void NavierStokesFractional<dim>::assemble_step1_system(const bool initial_step, const bool assemble_system)
{
    if(assemble_system)
        step1_matrix = 0;
    step1_rhs = 0;

    FEValues<dim> fe_values(*fe, *quadrature, update_values | update_gradients | 
												update_quadrature_points | update_JxW_values);

    FEFaceValues<dim> fe_face_values(*fe, *quadrature_face, update_values | update_normal_vectors | update_JxW_values);

    // usefull values referring to dofs and quadrature points
    const unsigned int dofs_per_cell = fe->n_dofs_per_cell();
    const unsigned int n_q_points = quadrature->size();
    const unsigned int n_q_face = quadrature_face->size();

    // for step 1 system we just require velocity
	const FEValuesExtractors::Vector velocities(0);

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<Tensor<1, dim>> present_velocity_values(n_q_points);
	std::vector<Tensor<2, dim>> present_velocity_gradients(n_q_points);

    std::vector<Tensor<1, dim>> old_velocity_values(n_q_points);
	std::vector<Tensor<2, dim>> old_velocity_gradients(n_q_points);

    std::vector<double> div_phi_u(dofs_per_cell);          // divergence of velocity
    std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);      // velocity
    std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell); // gradient of velocity

    for (const auto &cell : dof_handler.active_cell_iterators()) {
        if(!cell->is_locally_owned()) continue;

        fe_values.reinit(cell);

        local_matrix         = 0.0;
        local_rhs            = 0.0;

        fe_values[velocities].get_function_values(evaluation_point, present_velocity_values);
        fe_values[velocities].get_function_gradients(evaluation_point, present_velocity_gradients);

        // Same at old solution
        fe_values[velocities].get_function_values(old_solution, old_velocity_values);
        fe_values[velocities].get_function_gradients(old_solution, old_velocity_gradients);

        for (unsigned int q = 0; q < n_q_points; ++q) {
            for (unsigned int k = 0; k < dofs_per_cell; ++k) {
                div_phi_u[k] = fe_values[velocities].divergence(k, q);
                grad_phi_u[k] = fe_values[velocities].gradient(k, q);
                phi_u[k] = fe_values[velocities].value(k, q);
            };
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                if (assemble_matrix) {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                        // time derivative term (1/delta_t)<u,v>
                        local_matrix(i, j) += (1.0 / delta_t) * phi_u[i] * phi_u[j] * fe_values.JxW(q);

                        // Viscous term (theta * nu * <grad u, grad v>)
                        local_matrix(i, j) += viscosity * scalar_product(grad_phi_u[i], grad_phi_u[j]) * fe_values.JxW(q);

                        // Convective term (theta[< (grad u) u , v > + < (grad v) u, v >]), linearized 
                        // This linearizes the problem by using u_old as the advecting velocity
                        local_matrix(i, j) += (old_velocity_values[q] * grad_phi_u[j]) * phi_u[i] * fe_values.JxW(q);
                    };
                };
                local_rhs(i) += (old_velocity_values[q] / delta_t) * phi_u[i] * fe_values.JxW(q);
            };

        };
        cell->get_dof_indices(local_dof_indices);
			
        // this object here holds a list on constraint based on the fact wheter
        // this is the initial step or not.
        const AffineConstraints<double> &constraints_used = initial_step ? nonzero_constraints : zero_constraints;
        
        if (assemble_matrix) {
            constraints_used.distribute_local_to_global(local_matrix, local_rhs, local_dof_indices, step1_matrix, step1_rhs);
        } else {
            constraints_used.distribute_local_to_global(local_rhs, local_dof_indices, step1_rhs);
        }
    };
    step1_matrix.compress(VectorOperation::add);
    step1_rhs.compress(VectorOperation::add);
};

template <int dim>
void NavierStokesFractional<dim>::assemble_step2_system(const bool initial_step, const bool assemble_system)
{
    // The pressure matrix is constant if the mesh is static like in this case. 
    if(assemble_system) step2_matrix = 0;
    step2_rhs = 0;

    FEValues<dim> fe_values(*fe, *quadrature,
                            update_values | update_gradients | 
                            update_JxW_values | update_quadrature_points);

    const unsigned int dofs_per_cell = fe->n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature->size();

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    std::vector<double> div_intermediate_velocity(n_q_points);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;

        fe_values.reinit(cell);
        local_matrix = 0;
        local_rhs    = 0;

        // take the solution computed in the first step and apply the divergence to it
        fe_values[velocities].get_function_divergences(solution_tilde, div_intermediate_velocity);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                if(assemble_matrix)
                {
                    // if i is a velocity dof it will return 0 on both
                    const double         psi_p_i      = fe_values[pressure].value(i, q);
                    const Tensor<1, dim> grad_psi_p_i = fe_values[pressure].gradient(i, q);
    
                    // in fact we loop over j just if i is a pressure dof
                    if (fe->system_to_component_index(i).first == dim) 
                    {
                        for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                             // Only assemble if j is also a pressure dof
                             if (fe->system_to_component_index(j).first == dim)
                             {
                                 const Tensor<1, dim> grad_psi_p_j = fe_values[pressure].gradient(j, q);
                                 local_matrix(i, j) += scalar_product(grad_psi_p_i, grad_psi_p_j) * fe_values.JxW(q);
                             }
                        }
                        // This term is non-zero only if 'i' is a pressure DoF
                        local_rhs(i) -= (1.0 / delta_t) * div_intermediate_velocity[q] * psi_p_i * fe_values.JxW(q);
                    }              
                };
            };
        };
        // this object here holds a list on constraint based on the fact wheter
        // this is the initial step or not.
        const AffineConstraints<double> &constraints_used = initial_step ? nonzero_constraints : zero_constraints;
        
        if (assemble_matrix) {
            constraints_used.distribute_local_to_global(local_matrix, local_rhs, local_dof_indices, step2_matrix, step2_rhs);
        } else {
            constraints_used.distribute_local_to_global(local_rhs, local_dof_indices, step2_rhs);
        }
    }
    step2_matrix.compress(VectorOperation::add);
    step2_rhs.compress(VectorOperation::add);
}

template <int dim>
void NavierStokesFractional<dim>::assemble_step3_system(const bool initial_step, const bool assemble_matrix)
{
    if(assemble_matrix) step3_matrix = 0;
    step3_rhs = 0;

    FEValues<dim> fe_values(*fe, *quadrature,
                            update_values | update_gradients | 
                            update_JxW_values | update_quadrature_points);

    const unsigned int dofs_per_cell = fe->n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature->size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);
    
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Extractor to filter out pressure DoFs
    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    // we need this to store the solution of step 2
    std::vector<Tensor<1, dim>> pressure_gradients(n_q_points);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs = 0;

        fe_values[pressure].get_function_gradients(solution, pressure_gradients);
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const Tensor<1, dim> phi_i = fe_values[velocities].value(i, q);
                if(assemble_matrix)
                {
                    // Optimization: If phi_i is zero skip
                    if (fe->system_to_component_index(i).first == dim) continue;
    
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                        const Tensor<1, dim> phi_j = fe_values[velocities].value(j, q);
                        
                        if (fe->system_to_component_index(j).first == dim) continue;
    
                        // velocity_matrix = phi_i * phi_j
                        cell_matrix(i, j) += (phi_i * phi_j) * fe_values.JxW(q);
                    }
                }
                if (fe->system_to_component_index(i).first == dim) continue;

                // rhs = - delta_t * (grad p * v)
                cell_rhs(i) -= delta_t * (pressure_gradients[q] * phi_u_i) * dx;
            }
        }
        cell->get_dof_indices(local_dof_indices);
        const AffineConstraints<double> &constraints_used = initial_step ? nonzero_constraints : zero_constraints;
        
        if (assemble_matrix) {
            constraints_used.distribute_local_to_global(local_matrix, local_rhs, local_dof_indices, step3_matrix, step3_rhs);
        } else {
            constraints_used.distribute_local_to_global(local_rhs, local_dof_indices, step3_rhs);
        }
    }
    step3_matrix.compress(VectorOperation::add);
    step3_rhs.compress(VectorOperation::add);
}

template <int dim>
void NavierStokesFractional<dim>::solve_step1()
{
    SolverControl solver_control(10000, 1e-12, true);
    TrilinosWrappers::SolverGMRES solver(solver_control);

    // Preconditioner: AMG (ML or MueLu) is perfect for Advection-Diffusion
    TrilinosWrappers::PreconditionAMG preconditioner;
    TrilinosWrappers::PreconditionAMG::AdditionalData data;
    
    preconditioner.initialize(step1_matrix.block(0,0), data);

    // Solve strictly on Block 0
    solver.solve(step1_matrix.block(0,0), 
                 solution_tilde.block(0), 
                 step1_rhs.block(0), 
                 preconditioner);

    // set pressure DoFs to 0
    solution_tilde.block(1) = 0;
}

/** @brief we solve the step2 computing the pressure relative block of the solution 
*/
template <int dim>
void NavierStokesFractional<dim>::solve_step2()
{
    SolverControl solver_control(10000, 1e-12, true);
    TrilinosWrappers::SolverGMRES solver(solver_control);

    TrilinosWrappers::PreconditionAMG preconditioner;
    TrilinosWrappers::PreconditionAMG::AdditionalData data;
    
    preconditioner.initialize(step2_matrix, data);

    solver.solve(step2_matrix, 
                 this->solution.block(1), 
                 step2_rhs, 
                 preconditioner);

    // set pressure DoFs to 0
    solution_tilde.block(1) = 0;
}

/** @brief we solve the step3 calculating the projection of the step3_rhs 
*/
template <int dim>
void NavierStokesFractional<dim>::solve_step3()
{
    Vector<double> correction_vector;
    correction_vector.reinit(solution_tilde.block(0));

    SolverControl solver_control(1000, 1e-12);
    TrilinosWrappers::SolverCG solver(solver_control);
    TrilinosWrappers::PreconditionSSOR preconditioner; // since the system is symmetric
    preconditioner.initialize(velocity_mass_matrix.block(0,0));

    // Solve: M * correction = - dt * (grad p, v) computed in the assemble method
    solver.solve(step3_matrix.block(0,0), 
                 correction_vector, 
                 step3_rhs.block(0), 
                 preconditioner);

    solution.block(0) = solution_tilde.block(0); // copy u_tilde into the final solution vector
    
    // add the correction we calculated
    solution.block(0) += correction_vector; 
    
    zero_constraints.distribute(solution);
}

/** @brief Perform run time simulation of the solver running each step
 * for each timestep.
 */
template <int dim>
void NavierStokesFractional<dim>::run_time_simulation()
{
    setup_fractional_step_system();
    assemble_step2_system(true, true);
    assemble_step3_system(true, true);
    double time = 0.0;
    double time_step = this->time_step; // From base class

    while (time < this->end_time)
    {
        time += time_step;
        this->pcout << "Time step: " << time << std::endl;

        // assemble step1 --> matrix changes at each iteration becaue of convection term
        assemble_step1_system(false, true); 
        solve_step1(); // Result stored in 'solution_tilde'
        
        assemble_step2_system(false, false); 
        solve_step2(); // Result stored in 'this->solution' (just the pressure block)

        assemble_step3_system(false, false);
        solve_step3();

        old_solution = solution;

        this->output_results(time);
    }
}