#include "NavierStokesFractional.hpp"

#include <iomanip>
#include <cstdio>
#include <sstream>

namespace NavierStokes
{
    template <int dim>
    void NavierStokesFractional<dim>::setup_fractional_step_system()
    {
        this->initialize_system(); // call this method from the base class to initialize base members
        this->set_initial_condition();
        this->solution = 0;
    
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
    void NavierStokesFractional<dim>::assemble_step1_system(const bool initial_step, const bool assemble_matrix)
    {
        if(assemble_matrix)
            step1_matrix = 0;
        step1_rhs = 0;

        this->pcout << "\nAssembling Step 1 system." << "\n";

        FEValues<dim> fe_values(*this->fe, *this->quadrature, update_values | update_gradients | 
                                                    update_quadrature_points | update_JxW_values);
    
        FEFaceValues<dim> fe_face_values(*this->fe, *this->quadrature_face, update_values | update_normal_vectors | update_JxW_values);
    
        // usefull values referring to dofs and quadrature points
        const unsigned int dofs_per_cell = this->fe->n_dofs_per_cell();
        const unsigned int n_q_points = this->quadrature->size();
        const unsigned int n_q_face = this->quadrature_face->size();
    
        this->pcout << "dofs per cell:" << dofs_per_cell << "\n";
        this->pcout << "quadrature points:" << n_q_points << "\n";
        this->pcout << "quadrature face points:" << n_q_face << "\n";
        
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
    
        for (const auto &cell : this->dof_handler.active_cell_iterators()) {
            if(!cell->is_locally_owned()) continue;
    
            fe_values.reinit(cell);
    
            local_matrix         = 0.0;
            local_rhs            = 0.0;
    
            fe_values[velocities].get_function_values(this->solution, present_velocity_values);
            fe_values[velocities].get_function_gradients(this->solution, present_velocity_gradients);
    
            // Same at old solution
            fe_values[velocities].get_function_values(this->old_solution, old_velocity_values);
            fe_values[velocities].get_function_gradients(this->old_solution, old_velocity_gradients);
            
            for (unsigned int q = 0; q < n_q_points; ++q) {
                for (unsigned int k = 0; k < dofs_per_cell; ++k) {
                    div_phi_u[k] = fe_values[velocities].divergence(k, q);
                    grad_phi_u[k] = fe_values[velocities].gradient(k, q);
                    phi_u[k] = fe_values[velocities].value(k, q);
                };
                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    if (assemble_matrix) {
                        for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                            // time derivative term (1/this->delta_t)<u,v>
                            local_matrix(i, j) += (1.0 / this->delta_t) * phi_u[i] * phi_u[j] * fe_values.JxW(q);
    
                            // Viscous term (theta * nu * <grad u, grad v>)
                            local_matrix(i, j) += this->viscosity * scalar_product(grad_phi_u[i], grad_phi_u[j]) * fe_values.JxW(q);
    
                            // Convective term (theta[< (grad u) u , v > + < (grad v) u, v >]), linearized 
                            // This linearizes the problem by using u_old as the advecting velocity
                            local_matrix(i, j) += (old_velocity_values[q] * grad_phi_u[j]) * phi_u[i] * fe_values.JxW(q);
                        };
                    };
                    local_rhs(i) += (old_velocity_values[q] / this->delta_t) * phi_u[i] * fe_values.JxW(q);
                };
    
            };
            cell->get_dof_indices(local_dof_indices);
            // this object here holds a list on constraint based on the fact wheter
            // this is the initial step or not.
            const AffineConstraints<double> &constraints_used = initial_step ? this->nonzero_constraints : this->zero_constraints;
            
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
    void NavierStokesFractional<dim>::assemble_step2_system(const bool initial_step, const bool assemble_matrix)
    {
        // The pressure matrix is constant if the mesh is static like in this case. 
        if(assemble_matrix) step2_matrix = 0;
        step2_rhs = 0;

        this->pcout << "\nAssembling Step 2 system." << "\n";
    
        FEValues<dim> fe_values(*this->fe, *this->quadrature,
                                update_values | update_gradients | 
                                update_JxW_values | update_quadrature_points);
    
        const unsigned int dofs_per_cell = this->fe->n_dofs_per_cell();
        const unsigned int n_q_points    = this->quadrature->size();
    
        FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double>     local_rhs(dofs_per_cell);
    
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    
        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim);
    
        std::vector<double> div_intermediate_velocity(n_q_points);
    
        for (const auto &cell : this->dof_handler.active_cell_iterators())
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
                        if (this->fe->system_to_component_index(i).first == dim) 
                        {
                            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                            {
                                 // Only assemble if j is also a pressure dof
                                 if (this->fe->system_to_component_index(j).first == dim)
                                 {
                                     const Tensor<1, dim> grad_psi_p_j = fe_values[pressure].gradient(j, q);
                                     local_matrix(i, j) += scalar_product(grad_psi_p_i, grad_psi_p_j) * fe_values.JxW(q);
                                 }
                            }
                            // This term is non-zero only if 'i' is a pressure DoF
                            local_rhs(i) -= (1.0 / this->delta_t) * div_intermediate_velocity[q] * psi_p_i * fe_values.JxW(q);
                        }              
                    };
                };
            };
            // this object here holds a list on constraint based on the fact wheter
            // this is the initial step or not.
            const AffineConstraints<double> &constraints_used = initial_step ? this->nonzero_constraints : this->zero_constraints;
            
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

        this->pcout << "\nAssembling Step 3 system." << "\n";
    
        FEValues<dim> fe_values(*this->fe, *this->quadrature,
                                update_values | update_gradients | 
                                update_JxW_values | update_quadrature_points);
    
        const unsigned int dofs_per_cell = this->fe->n_dofs_per_cell();
        const unsigned int n_q_points    = this->quadrature->size();
    
        FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double>     local_rhs(dofs_per_cell);
        
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    
        // Extractor to filter out pressure DoFs
        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim);
    
        // we need this to store the solution of step 2
        std::vector<Tensor<1, dim>> pressure_gradients(n_q_points);
        std::vector<Tensor<1, dim>> tilde_values(n_q_points);
    
        for (const auto &cell : this->dof_handler.active_cell_iterators())
        {
            fe_values.reinit(cell);
            local_matrix = 0;
            local_rhs = 0;
    
            fe_values[pressure].get_function_gradients(this->solution, pressure_gradients);
            fe_values[velocities].get_function_values(this->solution_tilde, tilde_values);
            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    const Tensor<1, dim> phi_i = fe_values[velocities].value(i, q);
                    if(assemble_matrix)
                    {
                        // Optimization: If phi_i is zero skip
                        if (this->fe->system_to_component_index(i).first == dim) continue;
        
                        for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                            const Tensor<1, dim> phi_j = fe_values[velocities].value(j, q);
                            
                            if (this->fe->system_to_component_index(j).first == dim) continue;
        
                            // velocity_matrix = phi_i * phi_j
                            local_matrix(i, j) += (phi_i * phi_j) * fe_values.JxW(q);
                        }
                    }
                    if (this->fe->system_to_component_index(i).first == dim) continue;
                    local_rhs(i) += tilde_values[q] * phi_i * fe_values.JxW(q);
                    // rhs = - delta_t * (grad p * v)
                    local_rhs(i) -= this->delta_t * (pressure_gradients[q] * phi_i) * fe_values.JxW(q);
                }
            }
            cell->get_dof_indices(local_dof_indices);
            const AffineConstraints<double> &constraints_used = initial_step ? this->nonzero_constraints : this->zero_constraints;
            
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
        
        // Preconditioner: AMG (ML or MueLu) is perfect for Advection-Diffusion
        TrilinosWrappers::PreconditionAMG preconditioner;
        TrilinosWrappers::PreconditionAMG::AdditionalData data;
        SolverCG<TrilinosWrappers::MPI::Vector> cg(solver_control);
        
        preconditioner.initialize(step1_matrix.block(0,0), data);
    
        // Solve strictly on Block 0
        cg.solve(step1_matrix.block(0,0), 
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
        
        TrilinosWrappers::PreconditionAMG preconditioner;
        TrilinosWrappers::PreconditionAMG::AdditionalData data;
        SolverFGMRES<TrilinosWrappers::MPI::Vector> gmres(solver_control);
        
        preconditioner.initialize(step2_matrix.block(0, 0), data);

        TrilinosWrappers::MPI::Vector pressure;
        pressure.reinit(step2_rhs.block(0));
    
        gmres.solve(step2_matrix.block(0, 0), 
                     pressure, 
                     step2_rhs.block(0), 
                     preconditioner);

        this->solution.block(1) = pressure;
        // set pressure DoFs to 0
        solution_tilde.block(1) = 0;
    }
    
    /** @brief we solve the step3 calculating the projection of the step3_rhs 
    */
    template <int dim>
    void NavierStokesFractional<dim>::solve_step3()
    {
        TrilinosWrappers::MPI::Vector correction_vector;

        correction_vector.reinit(solution_tilde.block(0));
    
        SolverControl solver_control(1000, 1e-12);
        TrilinosWrappers::PreconditionSSOR preconditioner; // since the system is symmetric
        preconditioner.initialize(step3_matrix.block(0,0));
        SolverFGMRES<TrilinosWrappers::MPI::Vector> gmres(solver_control);
    
        // Solve: M * correction = - dt * (grad p, v) computed in the assemble method
        gmres.solve(step3_matrix.block(0,0), 
                     correction_vector, 
                     step3_rhs.block(0), 
                     preconditioner);
    
        this->solution.block(0) = solution_tilde.block(0); // copy u_tilde into the final solution vector
        
        // add the correction we calculated
        this->solution.block(0) += correction_vector; 
        
        this->zero_constraints.distribute(this->solution);
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
        double time_step = this->delta_t; // From base class
    
        while (time < this->T)
        {
            time += time_step;
            ++this->timestep_number;

            this->inlet_velocity_function->set_time(time);
			this->setup_boundaries();

			this->pcout << "\nTime step " << this->timestep_number << ", time = " << time << std::endl;
            this->old_solution = this->solution;
    
            // assemble step1 --> matrix changes at each iteration becaue of convection term
            assemble_step1_system(false, true); 
            solve_step1(); // Result stored in 'solution_tilde'
            
            assemble_step2_system(false, false); 
            solve_step2(); // Result stored in 'this->solution' (just the pressure block)
    
            assemble_step3_system(false, false);
            solve_step3();
            
    
            this->output_results();
        }
    }
	template class NavierStokesFractional<3>;

    // We explicitly tell the compiler to compile these classes (with these template parameters)
	template class NavierStokesFractional<2>;
};