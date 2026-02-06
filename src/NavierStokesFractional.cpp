#include "NavierStokesFractional.hpp"

#include <iomanip>
#include <cstdio>
#include <sstream>
#include <chrono>

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
    void NavierStokesFractional<dim>::setup_boundaries()
    {
        this->nonzero_constraints.clear();
        
        DoFTools::make_hanging_node_constraints(this->dof_handler, this->nonzero_constraints);
        
        // Define masks: We only want to constrain VELOCITY components here.
        FEValuesExtractors::Vector velocity(0);
        ComponentMask velocity_mask = this->fe->component_mask(velocity);
        
        {
            std::map<types::boundary_id, const Function<dim> *> boundary_functions;
            Functions::ZeroFunction<dim> zero_function(dim + 1);
            boundary_functions[0] = this->inlet_velocity_function.get();

            // ID 2, 3: Walls/Obstacle (Zero Velocity)
            boundary_functions[2] = &zero_function; 
            boundary_functions[3] = &zero_function; 
            
            // Note: ID 1 (Outlet) is OMITTED. This creates the "Do-Nothing" condition.
            
            VectorTools::interpolate_boundary_values(this->dof_handler,
                boundary_functions,
                this->nonzero_constraints,
                velocity_mask);
            }
        this->nonzero_constraints.close();
        // clear zero constraints now    
        this->zero_constraints.clear();
        DoFTools::make_hanging_node_constraints(this->dof_handler, this->zero_constraints);
        {
            std::map<types::boundary_id, const Function<dim> *> boundary_functions;
            Functions::ZeroFunction<dim> zero_function(dim + 1);

            // ID 0: Inlet -> ZERO (Don't change velocity during correction)
            boundary_functions[0] = &zero_function; 
            
            // ID 2, 3: Walls -> ZERO
            boundary_functions[2] = &zero_function;
            boundary_functions[3] = &zero_function;

            VectorTools::interpolate_boundary_values(this->dof_handler,
                                                    boundary_functions,
                                                    this->zero_constraints,
                                                    velocity_mask);
        }

        this->zero_constraints.close();

        pressure_constraints.clear(); // Make sure you added this object to your class!
        DoFTools::make_hanging_node_constraints(this->dof_handler, pressure_constraints);
        
        // Extract strictly the pressure part of the system
        std::vector<bool> pressure_components(dim + 1, false);
        pressure_components[dim] = true; 
        ComponentMask pressure_mask(pressure_components);

        IndexSet pressure_dofs = DoFTools::extract_dofs(this->dof_handler, pressure_mask);
        
        // Only Processor 0 adds the line
        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
            if (pressure_dofs.n_elements() > 0) {
                // Pick the very first pressure DoF we own
                const auto first_pressure_dof = pressure_dofs.nth_index_in_set(0);
                
                this->pressure_constraints.add_line(first_pressure_dof);
                this->pressure_constraints.set_inhomogeneity(first_pressure_dof, 0.0);
            }
        }
        pressure_constraints.close();
    }
    
    template<int dim>
    void NavierStokesFractional<dim>::assemble_step1_system(const bool assemble_matrix)
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
        
        // for step 1 system we just require velocity
        const FEValuesExtractors::Vector velocities(0);

        FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> local_rhs(dofs_per_cell);
    
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    
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
    
                            // This linearizes the problem by using u_old as the advecting velocity
                            local_matrix(i, j) += (grad_phi_u[j] * old_velocity_values[q]) * phi_u[i] * fe_values.JxW(q);
                        };
                    };
                    local_rhs(i) += (old_velocity_values[q] / this->delta_t) * phi_u[i] * fe_values.JxW(q);
                };
    
            };
            // boundary conditions
			if(cell->at_boundary()){
				for(size_t f = 0; f < cell->n_faces();++f){
					// apply that to the outlet boundary where the id == 2 --> look gmsh to be sure
					if(cell->face(f)->at_boundary() && cell->face(f)->boundary_id() == 2){
						fe_face_values.reinit(cell, f);

						for (size_t q = 0; q < n_q_face; ++q){
							for (size_t i = 0; i < dofs_per_cell; ++i){
									local_rhs(i) += -this->p_out * 
										scalar_product(fe_face_values.normal_vector(q),
										fe_face_values[velocities].value(i, q)) * fe_face_values.JxW(q);
								}
						}
					}
				}
            }
            cell->get_dof_indices(local_dof_indices);
            // this object here holds a list on constraint based on the fact wheter
            // this is the initial step or not.
            const AffineConstraints<double> &constraints_used = this->nonzero_constraints;
            
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
    void NavierStokesFractional<dim>::assemble_step2_system(const bool assemble_matrix)
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
                        }              
                    };
                    if (this->fe->system_to_component_index(i).first == dim) 
                    {
                        const double psi_p_i = fe_values[pressure].value(i, q);
                        // -(1/dt) * (div u_tilde, q)
                        local_rhs(i) -= (1.0 / this->delta_t) * div_intermediate_velocity[q] * psi_p_i * fe_values.JxW(q);
                    }
                };
            };
            // this object here holds a list on constraint based on the fact wheter
            // this is the initial step or not.
            const AffineConstraints<double> &constraints_used = pressure_constraints; // I consider zero constraints for step2 (and 3 aswell)
            
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
    void NavierStokesFractional<dim>::assemble_step3_system(const bool assemble_matrix)
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
                    // rhs = - delta_t * (grad p * v)
                    local_rhs(i) -= this->delta_t * (pressure_gradients[q] * phi_i) * fe_values.JxW(q);
                }
            }
            cell->get_dof_indices(local_dof_indices);
            const AffineConstraints<double> &constraints_used = this->zero_constraints;
            
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
        this->pcout << step1_rhs.l2_norm() << "\n";
        auto start = std::chrono::high_resolution_clock::now();

        SolverControl solver_control(10000, 1e-10, true);

        // Preconditioner: AMG (ML or MueLu) is perfect for Advection-Diffusion
        TrilinosWrappers::PreconditionILU preconditioner;
        TrilinosWrappers::PreconditionILU::AdditionalData data;
        if(this->mpi_size == 1) 
            data.overlap = 0; // Increase if using many MPI processes
        else
            data.overlap = this->mpi_size;

        data.ilu_fill = 1;
        data.ilu_atol = 1e-4;
        data.ilu_rtol = 1.01; // setting the values suggested on the documentations.
        SolverFGMRES<TrilinosWrappers::MPI::Vector> gmres(solver_control);

        preconditioner.initialize(step1_matrix.block(0,0), data);
    
        // Solve strictly on Block 0
        gmres.solve(step1_matrix.block(0,0), 
                     solution_tilde.block(0), 
                     step1_rhs.block(0), 
                     preconditioner);
    
        this->nonzero_constraints.distribute(solution_tilde); // distribute the constraints to the partial solution
        // set pressure DoFs to 0
        solution_tilde.block(1) = 0;
        this->pcout << "Solution norm: " << this->solution.l2_norm() << "\n";
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        this->pcout << "Step 2 duration is: " << diff.count() << "\n";

    }

    /** @brief we solve the step2 computing the pressure relative block of the solution 
    */
    template <int dim>
    void NavierStokesFractional<dim>::solve_step2()
    {
        SolverControl solver_control(10000, 1e-12, true);
        auto start = std::chrono::high_resolution_clock::now();
        
        TrilinosWrappers::PreconditionAMG preconditioner;
        TrilinosWrappers::PreconditionAMG::AdditionalData data;
        data.elliptic = true; // enables optimization for elliptic problems (many parameters could be set here for optimization)

        SolverCG<TrilinosWrappers::MPI::Vector> cg(solver_control); // since the matrix is SPD
        
        preconditioner.initialize(step2_matrix.block(1, 1), data);

        TrilinosWrappers::MPI::Vector pressure;
        pressure.reinit(step2_rhs.block(1));
    
        cg.solve(step2_matrix.block(1, 1), 
                     pressure, 
                     step2_rhs.block(1), 
                     preconditioner);

        this->solution.block(1) = pressure;
        // set pressure DoFs to 0
        solution_tilde.block(1) = 0;
        pressure_constraints.distribute(this->solution);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        this->pcout << "Step 2 duration is: " << diff.count() << "\n";
    }

    /** @brief we solve the step3 calculating the projection of the step3_rhs 
    */
    template <int dim>
    void NavierStokesFractional<dim>::solve_step3()
    {
        TrilinosWrappers::MPI::Vector correction_vector;
        auto start = std::chrono::high_resolution_clock::now();

        correction_vector.reinit(solution_tilde.block(0));

        // using Jacobi since the system is very simple and SPD so no need of advanced preconditioners
        SolverControl solver_control(1000, 1e-12);
        TrilinosWrappers::PreconditionJacobi preconditioner;
        preconditioner.initialize(step3_matrix.block(0,0));

        SolverCG<TrilinosWrappers::MPI::Vector> cg(solver_control);
    
        // Solve: M * correction = - dt * (grad p, v) computed in the assemble method
        cg.solve(step3_matrix.block(0,0), 
                     correction_vector, 
                     step3_rhs.block(0), 
                     preconditioner);
                     
        this->zero_constraints.distribute(correction_vector); // we enforce zero constraints on the correction
        this->solution.block(0) = solution_tilde.block(0); // copy u_tilde into the final solution vector
        
        // add the correction we calculated
        this->solution.block(0) += correction_vector; 
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        this->pcout << "Step 3 duration is: " << diff.count() << "\n";
    }

    /** @brief Perform run time simulation of the solver running each step
     * for each timestep.
     */
    template <int dim>
    void NavierStokesFractional<dim>::run_time_simulation()
    {
        setup_fractional_step_system();
        assemble_step2_system(true);
        assemble_step3_system(true);
        double time = 0.0;
        double time_step = this->delta_t; // From base class
        this->pcout << "delta t: " << time_step << "\n";

        while (time < this->T)
        {
            time += time_step;
            ++this->timestep_number;

            this->inlet_velocity_function->set_time(time);
			setup_boundaries();

			this->pcout << "\nTime step " << this->timestep_number << ", time = " << time << std::endl;
            this->old_solution = this->solution;
    
            // assemble step1 --> matrix changes at each iteration becaue of convection term
            assemble_step1_system(true); 
            solve_step1(); // Result stored in 'solution_tilde'
            
            assemble_step2_system(false); 
            solve_step2(); // Result stored in 'this->solution' (just the pressure block)
    
            assemble_step3_system(false);
            solve_step3();
            
            this->output_results();
        }
    }
	template class NavierStokesFractional<3>;

    // We explicitly tell the compiler to compile these classes (with these template parameters)
	template class NavierStokesFractional<2>;
};