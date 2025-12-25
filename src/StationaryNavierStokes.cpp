#include <fstream>
#include <iomanip>
#include "StationaryNavierStokes.hpp"
#include "./preconditioners/BlockPrecondtioner.h"
#include "./preconditioners/BlockSchurPreconditioner.hpp"
#include "./preconditioners/BlockTriangularPrecondition.hpp"
#include "./preconditioners/PreconditionIdentity.h"


namespace NavierStokes{
	
	template <int dim>
	void StationaryNavierStokes<dim>::initialize_system()
	{
		{
			pcout << "Initializing the mesh" << std::endl;

			Triangulation<dim> mesh_serial;

			GridIn<dim> grid_in;
			grid_in.attach_triangulation(mesh_serial);

			std::ifstream grid_in_file(mesh_file_name);
			grid_in.read_msh(grid_in_file);

			GridTools::partition_triangulation(mpi_size, mesh_serial);
			const auto construction_data = TriangulationDescription::Utilities::
			create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
			mesh.create_triangulation(construction_data);
		}
		// Initialization of the finite element space
		const FE_SimplexP<dim> fe_scalar_velocity(degree_velocity);
		const FE_SimplexP<dim> fe_scalar_pressure(degree_pressure);

		// velocity and pressure will be considered as blocks and not components
		fe = std::make_unique<FESystem<dim>>(fe_scalar_velocity,
											 dim,
											 fe_scalar_pressure,
											 1);

		// initialize quadrature. We need this instead of dealII implementation as we 
		// are using triangulations and gmsh.
		quadrature = std::make_unique<QGaussSimplex<dim>>(fe->degree + 1);
		quadrature_face = std::make_unique<QGaussSimplex<dim - 1>>(fe->degree + 1);
		pcout << "  Quadrature points per cell = " << quadrature->size()
		<< std::endl;
		pcout << "  Quadrature points per face = " << quadrature_face->size() << std::endl;

		// SETUP DOFS AND BOUNDARIES
     	setup_dofs();
		setup_boundaries();
		
		pcout << "  Initializing the sparsity pattern" << std::endl;

		Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
		for (unsigned int c = 0; c < dim + 1; ++c)
		{
			for (unsigned int d = 0; d < dim + 1; ++d)
			{
				if (c == dim && d == dim) // pressure-pressure term do not appear in the equations
					coupling[c][d] = DoFTools::always;
				else // other combinations
					coupling[c][d] = DoFTools::always;
			}
		}

		sparsity_pattern.reinit(block_owned_dofs, MPI_COMM_WORLD);
		DoFTools::make_sparsity_pattern(dof_handler, coupling, sparsity_pattern);
		sparsity_pattern.compress();

		// We also build a sparsity pattern for the pressure mass matrix.
		for (unsigned int c = 0; c < dim + 1; ++c)
		{
			for (unsigned int d = 0; d < dim + 1; ++d)
			{
				if (c == dim && d == dim) // pressure-pressure term
				coupling[c][d] = DoFTools::always;
				else // other combinations
				coupling[c][d] = DoFTools::none;
			}
		}
		TrilinosWrappers::BlockSparsityPattern sparsity_pressure_mass(block_owned_dofs,
																	 MPI_COMM_WORLD);
		DoFTools::make_sparsity_pattern(dof_handler,
										coupling,
										sparsity_pressure_mass);
		sparsity_pressure_mass.compress();

		// code taken from Bucelli's labs
		pcout << "  Initializing the system matrix" << std::endl;
		system_matrix.reinit(sparsity_pattern);
		pressure_mass.reinit(sparsity_pressure_mass);
	
		pcout << "  Initializing the solution vector" << std::endl;
		present_solution.reinit(block_owned_dofs, MPI_COMM_WORLD);
		newton_update.reinit(block_owned_dofs, MPI_COMM_WORLD);
		evaluation_point.reinit(block_owned_dofs, MPI_COMM_WORLD);
	
		pcout << "  Initializing the system right-hand side" << std::endl;
		system_rhs.reinit(block_owned_dofs, MPI_COMM_WORLD);
		
		pcout << "  Initializing the solution vector" << std::endl;
		solution_owned.reinit(block_owned_dofs, MPI_COMM_WORLD);
		solution.reinit(block_owned_dofs, block_relevant_dofs, MPI_COMM_WORLD);
	}

	/** @brief Function that setups the Dof handler for the Stationary Navier Stokes solver. 
	 */
	template<int dim>
	void StationaryNavierStokes<dim>::setup_dofs()
	{
		pcout << "Initializing the DoF handler." << std::endl;

		dof_handler.reinit(mesh);
		dof_handler.distribute_dofs(*fe);

		// enforce a specific ordering for the dofs. (I need this order to be respected)
		// block component has values 0 and 1 where:
		//     - 0 means velocity dof
		//     - 1 means pressure dof
		std::vector<unsigned int> block_component(dim + 1, 0);
		block_component[dim] = 1;
		DoFRenumbering::component_wise(dof_handler, block_component);

		locally_owned_dofs = dof_handler.locally_owned_dofs();
		locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

		// now get the block dofs for velocity and pressure
		std::vector<types::global_dof_index> dofs_per_block = DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
		const unsigned int n_u = dofs_per_block[0]; 
		const unsigned int n_p = dofs_per_block[1]; 

		block_owned_dofs.resize(2);
		block_relevant_dofs.resize(2);

		// get_view gives back a subset of elements of a vector
		block_owned_dofs[0] = locally_owned_dofs.get_view(0, n_u);
		block_relevant_dofs[0] = locally_relevant_dofs.get_view(0, n_u);
		
		// logic here is to take n_p elements from the n_u index and assign it to the second block
		block_owned_dofs[1] = locally_owned_dofs.get_view(n_u, n_u + n_p);
		block_relevant_dofs[1] = locally_relevant_dofs.get_view(n_u, n_u + n_p);
	}

	/** @brief Setup boundary conditions for each of the surfaces (inlet, walls, outlet)
	 */
	template<int dim>
	void StationaryNavierStokes<dim>::setup_boundaries()
	{
		pcout << "Setup Boundaries." << std::endl;

		nonzero_constraints.clear();
    	DoFTools::make_hanging_node_constraints(dof_handler, nonzero_constraints);

		std::map<types::boundary_id, const Function<dim> *> boundary_functions;
		Functions::ZeroFunction<dim> zero_function(dim + 1);
		FEValuesExtractors::Vector velocity(0);
		ComponentMask velocity_mask = fe->component_mask(velocity);
								
		boundary_functions[0] = &inlet_velocity; // Inlet velocity
											     // no velocity set at the outlet	
		boundary_functions[2] = &zero_function;  // Walls
		boundary_functions[3] = &zero_function;  // Obstacle

		VectorTools::interpolate_boundary_values(dof_handler,
												boundary_functions,
												nonzero_constraints,
												velocity_mask);
		
		nonzero_constraints.close();
		zero_constraints.clear();
		DoFTools::make_hanging_node_constraints(dof_handler, zero_constraints);

		// Apply u = 0.0 to all Dirichlet boundaries
		{			
			// The Inlet is now set to zero cause we dont want newton to update it.
			// We don't want to change the inlet velocity during an update.
			boundary_functions[0] = &zero_function; 
			
			// Walls and Obstacles are also zero (obviously)
			boundary_functions[2] = &zero_function;
			boundary_functions[3] = &zero_function;

			// we assign these 
			VectorTools::interpolate_boundary_values(dof_handler,
													boundary_functions,
													zero_constraints,
													velocity_mask);
		}
		zero_constraints.close();

		// Fix pressure at one DoF to remove null space 
		// Maybe we need mean zero condition? We are not using Neummann BC, so maybe
		// we should indeed use L2_0 mean zero condition.
		std::vector<bool> pressure_components(dim + 1, false);
		pressure_components[dim] = true; // pressure is last component
		ComponentMask pressure_mask(pressure_components);

		IndexSet pressure_dofs = DoFTools::extract_dofs(dof_handler, pressure_mask);
		if (pressure_dofs.n_elements() > 0) {
		    const auto first_pressure_dof = pressure_dofs.nth_index_in_set(0);
		    nonzero_constraints.add_line(first_pressure_dof);
		    nonzero_constraints.set_inhomogeneity(first_pressure_dof, 0.0);
		    zero_constraints.add_line(first_pressure_dof);
		    zero_constraints.set_inhomogeneity(first_pressure_dof, 0.0);
		}
	}

	
	/** @brief Assembles the whole system. (too lazy) */
	template <int dim>
	void StationaryNavierStokes<dim>::assemble(const bool initial_step, const bool assemble_matrix)
	{
		
		if (assemble_matrix) system_matrix = 0;
		system_rhs = 0;
		pressure_mass = 0;
	
		FEValues<dim> fe_values(*fe, *quadrature, update_values | update_gradients | 
												update_quadrature_points | update_JxW_values);
	
		FEFaceValues<dim> fe_face_values(*fe, *quadrature_face, update_values | update_normal_vectors | 
													update_JxW_values);
		
		
		// usefull values referring to dofs and quadrature points
		const unsigned int dofs_per_cell = fe->n_dofs_per_cell();
		const unsigned int n_q_points = quadrature->size();
 		const unsigned int n_q_face      = quadrature_face->size();

		const FEValuesExtractors::Vector velocities(0);
    	const FEValuesExtractors::Scalar pressure(dim);
	
		FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
		Vector<double>     local_rhs(dofs_per_cell);
		FullMatrix<double> cell_pressure_mass_matrix(dofs_per_cell, dofs_per_cell);
	
		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
	
		// vecotrs used to store past values at each quadrature point
		std::vector<Tensor<1, dim>> present_velocity_values(n_q_points);
		std::vector<Tensor<2, dim>> present_velocity_gradients(n_q_points);
		std::vector<double> present_pressure_values(n_q_points);
	
		// vectors used to store values of the test functions
		std::vector<double> div_phi_u(dofs_per_cell);  			// divergence of velocity
		std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);       // velocity 
		std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);  // gradient of velocity
		std::vector<double> phi_p(dofs_per_cell);               // pressure
		

		for (const auto &cell : dof_handler.active_cell_iterators()) {
			if(!cell->is_locally_owned()) continue;

			fe_values.reinit(cell);
	
			local_matrix         = 0.0;
			cell_pressure_mass_matrix = 0.0;
			local_rhs            = 0.0;
	

			// We need to know the values and the gradient of velocity on quadrature nodes (explained by Bucelli)
			fe_values[velocities].get_function_values(evaluation_point, present_velocity_values);
			fe_values[velocities].get_function_gradients(evaluation_point, present_velocity_gradients);
			fe_values[pressure].get_function_values(evaluation_point, present_pressure_values);
	
			for (const auto& tensor : present_velocity_gradients) {
			if (std::isinf(tensor.norm())) {
				pcout << "TROVATO GRADIENTE INFINITO NELLA CELLA: " << cell->active_cell_index() << std::endl;
				// Puoi anche stampare i vertici della cella per vederla
				}
			}
			for (unsigned int q = 0; q < n_q_points; ++q) {
				for (unsigned int k = 0; k < dofs_per_cell; ++k) {
					div_phi_u[k] = fe_values[velocities].divergence(k, q);
					grad_phi_u[k] = fe_values[velocities].gradient(k, q);
					phi_u[k] = fe_values[velocities].value(k, q);
					phi_p[k] = fe_values[pressure].value(k, q);
				}
				for (unsigned int i = 0; i < dofs_per_cell; ++i) {
					if (assemble_matrix) {
						for (unsigned int j = 0; j < dofs_per_cell; ++j) {
							local_matrix(i, j) += (viscosity * scalar_product(grad_phi_u[i], grad_phi_u[j])
													 + phi_u[i] * (present_velocity_gradients[q] * phi_u[j]) 
													+ phi_u[i] * (grad_phi_u[j] * present_velocity_values[q])
													- div_phi_u[i] * phi_p[j] - phi_p[i] * div_phi_u[j] 
													+ gamma * div_phi_u[i] * div_phi_u[j]) * fe_values.JxW(q); // here there was this term here: phi_p[i] * phi_p[j] used in pressure matrix
							
							// added this, exactly how bucelli implemented it. Dont know if it's mathematically correct tough
							cell_pressure_mass_matrix(i, j) += phi_p[i] * phi_p[j] * fe_values.JxW(q);
						}
					}
					double present_velocity_divergence = trace(present_velocity_gradients[q]);
					local_rhs(i) += (-viscosity * scalar_product(grad_phi_u[i], present_velocity_gradients[q]) 
									- phi_u[i] * (present_velocity_gradients[q] * present_velocity_values[q]) 
									+ div_phi_u[i] * present_pressure_values[q]
									+ phi_p[i] * present_velocity_divergence
									- gamma * div_phi_u[i] * present_velocity_divergence) 
									* fe_values.JxW(q);
				}
			}

			// boundary conditions
			if(cell->at_boundary()){
				for(size_t f = 0; f < cell->n_faces();++f){
					// apply that to the outlet boundary where the id == 2 --> look gmsh to be sure
					if(cell->face(f)->at_boundary() && cell->face(f)->boundary_id() == 2){
						fe_face_values.reinit(cell, f);

						for (size_t q = 0; q < n_q_face; ++q){
							for (size_t i = 0; i < dofs_per_cell; ++i){
									local_rhs(i) += -p_out * 
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
			const AffineConstraints<double> &constraints_used = initial_step ? nonzero_constraints : zero_constraints;
			if (assemble_matrix) {
				constraints_used.distribute_local_to_global(local_matrix, local_rhs, local_dof_indices, system_matrix, system_rhs);
			} else {
				constraints_used.distribute_local_to_global(local_rhs, local_dof_indices, system_rhs);
			}
			if (assemble_matrix) pressure_mass.add(local_dof_indices, cell_pressure_mass_matrix);
		}
		// I dont know if here we need a check on assemble matrix for system and pressure_mass
		system_matrix.compress(VectorOperation::add);
		pressure_mass.compress(VectorOperation::add);
		system_rhs.compress(VectorOperation::add);
	}
	
	template <int dim>
	void StationaryNavierStokes<dim>::assemble_system(const bool initial_step)
	{
		assemble(initial_step, true);
	}
	
	template <int dim>
	void StationaryNavierStokes<dim>::assemble_rhs(const bool initial_step)

	template <int dim>
	void StationaryNavierStokes<dim>::solve(const bool initial_step)
	{
		// as before we define contraints bsased on the iteration we're on
		const AffineConstraints<double> &constraints_used = initial_step ? nonzero_constraints : zero_constraints;
	
		// initialize object for solving the system
		SolverControl solver_control(system_matrix.m(), 1e-4 * system_rhs.l2_norm(), true);
		SolverFGMRES<TrilinosWrappers::MPI::BlockVector> gmres(solver_control);
		
		/*
		// initialize ILU preconditioner with the pressure mass matrix we derived in the assemble() function
		TrilinosWrappers::PreconditionILU pmass_preconditioner;
		pmass_preconditioner.initialize(pressure_mass.block(0,0), 
					TrilinosWrappers::PreconditionILU::AdditionalData());

		// initialize BlockShurPreconditioner passing the previously computed pmass precondtioner;
		const BlockSchurPreconditioner<TrilinosWrappers::PreconditionILU> preconditioner(gamma, viscosity, system_matrix, pressure_mass, pmass_preconditioner);
		*/
		
		PreconditionBlockTriangular preconditioner;
  		preconditioner.initialize(system_matrix.block(0, 0),
                            		pressure_mass.block(1, 1),
                            		system_matrix.block(1, 0));

		// solve using the Shur Preconditioner
		gmres.solve(system_matrix, newton_update, system_rhs, preconditioner);
		pcout << "FGMRES steps: " << solver_control.last_step() << std::endl;
		constraints_used.distribute(newton_update);

		solution = newton_update; // update owned and ghost dofs
	}

	/** @brief We use the Newton method in order to solve a nonlinear system like
	 * 	this one 
	 */
	template <int dim>
	void StationaryNavierStokes<dim>::newton_iteration(const double tolerance, const unsigned int max_n_line_searches, const bool is_initial_step, const bool output_result)
	{
		bool first_step = is_initial_step;
		unsigned int line_search_n = 0;
		double last_res = 1.0;
		double current_res = 1.0;

		// main loop here that controls the nonlinear solver given a tolerance
		while ((first_step || (current_res > tolerance)) && line_search_n < max_n_line_searches) {
			if (first_step) {
				// initialize and assemble the system in the first iter
				initialize_system();
				evaluation_point = present_solution;
				assemble_system(first_step);

				solve(first_step);
				present_solution = newton_update;
				nonzero_constraints.distribute(present_solution);
				first_step = false; // ensure we do not call this routine again 
				evaluation_point = present_solution;
				assemble_rhs(first_step);
				current_res = system_rhs.l2_norm();
				pcout << "The residual of initial guess is " << current_res << std::endl;
				last_res = current_res;
			} else {
				evaluation_point = present_solution;
				assemble_system(first_step);
				solve(first_step);

				// interesting backtracking approach, instead of taking a fixed alpha for the newton iter it halves
				// it each time				
				for (double alpha = 1.0; alpha > 1e-5; alpha *= 0.5) {
					evaluation_point = present_solution;
					evaluation_point.add(alpha, newton_update);
					solution = evaluation_point; // we must pass the information to ghost dofs aswell

					nonzero_constraints.distribute(evaluation_point);

					assemble_rhs(first_step);
					current_res = system_rhs.l2_norm();
					pcout << "  alpha: " << std::setw(10) << alpha << std::setw(0) << "  residual: " << current_res << std::endl;
					if (current_res < last_res)
						break;
				}
				{
					present_solution = evaluation_point;
					pcout << "  number of line searches: " << line_search_n << "  residual: " << current_res << std::endl;
					last_res = current_res;
				}
				++line_search_n; // increment line search number
			}
		}
		// output result decides wheter to store or not results.
		if (output_result) {
			output_results();
			// process_solution(); no need for now of this function call
		}
	}
	
	/** @brief Function that computes the initial guess when the Reynolds number is over 1000
	 * I guess we will not need this as in the paper it is written we are just working with
	 * Reynolds number smaller than 200.
	 */
	template <int dim>
	void StationaryNavierStokes<dim>::compute_initial_guess(double step_size)
	{
		const double target_Re = 1.0 / viscosity;
		bool is_initial_step = true;
		for (double Re = 1000.0; Re < target_Re; Re = std::min(Re + step_size, target_Re)) {
			viscosity = 1.0 / Re;
			pcout << "Searching for initial guess with Re = " << Re << std::endl;
			newton_iteration(1e-12, 50, is_initial_step, false);
			is_initial_step = false;
		}
	}
	
	
	/** @brief Function that just prints the output (pretty similar to Bucelli's) 
	*/
	template <int dim>
	void StationaryNavierStokes<dim>::output_results() const
	{
	    pcout << "===============================================" << std::endl;

		// vector of strings with dim element as "velocity" and last element "pressure"
		std::vector<std::string> solution_names(dim, "velocity");
		solution_names.emplace_back("pressure");
	
		std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
		data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
	
		DataOut<dim> data_out;
  		data_out.add_data_vector(dof_handler, 
								solution, 
								solution_names, 
								data_component_interpretation);

		std::vector<unsigned int> partition_int(mesh.n_active_cells());
		GridTools::get_subdomain_association(mesh, partition_int);
		const Vector<double> partitioning(partition_int.begin(), partition_int.end());
		data_out.add_data_vector(partitioning, "partitioning");
		data_out.build_patches();
	
		// here to insert correct ReyNolds Number aswell REMEMBER THIS 
		const std::string output_file_name = std::to_string(static_cast<int>(std::round(1.0 / viscosity))) + "Re-SNS_Solution";
		data_out.write_vtu_with_pvtu_record("../results/",
											output_file_name,
											0,
											MPI_COMM_WORLD);
	
		pcout << "Output written to " << output_file_name << "." << std::endl;
		pcout << "===============================================" << std::endl;
	}
	
	/** @brief Function that it samples the velocity profile along the vertical centerline
	 *  of the domain and writes it to a text file. Usefull for testing and validation purposes.
	 */
	template <int dim>
	void StationaryNavierStokes<dim>::process_solution()
	{
		// here Reynolds number to insert of course REMEMBER THIS 
		std::ofstream f(std::to_string(1.0 / viscosity) + "-line-" + ".txt");
		f << "# y u_x u_y" << std::endl;
		Point<dim> p;
		p[0] = 0.5;
		p[1] = 0.5;
		f << std::scientific;
	
		// samples specifically 101 points
		for (unsigned int i = 0; i <= 100; ++i) {
			p[dim - 1] = i / 100.0;
			Vector<double> tmp_vector(dim + 1);
	
			// given p it calculates the value of the present solution and stores it in tmp vector
			VectorTools::point_value(dof_handler, present_solution, p, tmp_vector);
			f << p[dim - 1];
			for (int j = 0; j < dim; ++j)
				f << ' ' << tmp_vector(j);
			f << std::endl;
		}
	}
	
	/** @brief Method that actually run the whole method calling the newton iteration.  
	 *  With really high Reynolds Number (not the case for us I guess) it also computes
	 *  an initial guess calling the "compute_initial_guess" function.
	*/
	template <int dim>
	void StationaryNavierStokes<dim>::run()
	{
		const double Re = 1.0 / viscosity;
		if (Re > 1000.0) {
			pcout << "Searching for initial guess ..." << std::endl;
			const double step_size = 2000.0;
			compute_initial_guess(step_size);
			pcout << "Found initial guess." << std::endl;
			pcout << "Computing solution with target Re = " << Re << std::endl;
			viscosity = 1.0 / Re;
			newton_iteration(1e-12, 50, false, true);
		} else {
			newton_iteration(1e-12, 50, true, true);
		}
	}
	
	// Explicit instantiation for dim=2
	template class StationaryNavierStokes<3>;	
};
