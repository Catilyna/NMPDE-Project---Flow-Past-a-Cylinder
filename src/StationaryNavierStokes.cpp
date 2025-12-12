#include "StationaryNavierStokes.hpp"
#include "BoundaryValues.h"
#include <fstream>
#include <iomanip>
#include "./preconditioners/BlockPrecondtioner.h"
#include "./preconditioners/BlockSchurPreconditioner.h"
#include "./preconditioners/BlockTriangularPrecondition.hpp"
#include "./preconditioners/PreconditionIdentity.h"
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/utilities.h>


namespace NavierStokes{
	
	template <int dim>
	void StationaryNavierStokes<dim>::initialize_system()
	{
		// part of initializing the mesh + dof handler is missing as I wait for bucelli to
		// load full code of the previous lab

		
		pcout << "  Initializing the sparsity pattern" << std::endl;

		Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
		for (unsigned int c = 0; c < dim + 1; ++c)
		{
			for (unsigned int d = 0; d < dim + 1; ++d)
			{
				if (c == dim && d == dim) // pressure-pressure term do not appear in the equations
				coupling[c][d] = DoFTools::none;
				else // other combinations
				coupling[c][d] = DoFTools::always;
			}
		}

		TrilinosWrappers::BlockSparsityPattern sparsity(block_owned_dofs,
                                                        MPI_COMM_WORLD);
		DoFTools::make_sparsity_pattern(dof_handler, coupling, sparsity);
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
		present_solution.reinit(dofs_per_block);
	
		newton_update.reinit(dofs_per_block);
	
		pcout << "  Initializing the system right-hand side" << std::endl;
		system_rhs.reinit(dofs_per_block, MPI_COMM_WORLD);
		pcout << "  Initializing the solution vector" << std::endl;
		solution_owned.reinit(block_owned_dofs, MPI_COMM_WORLD);
		solution.reinit(block_owned_dofs, block_relevant_dofs, MPI_COMM_WORLD);
	}
	
	/** @brief Assembles the whole system. (too lazy) */
	template <int dim>
	void StationaryNavierStokes<dim>::assemble(const bool initial_step, const bool assemble_matrix)
	{
		if (assemble_matrix)
			system_matrix = 0;
		system_rhs = 0;
	
		const QGauss<dim> quadrature_formula(degree + 2);
		FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_quadrature_points | update_JxW_values | update_gradients);
	
		// usefull values referring to dofs and quadrature points
		const unsigned int dofs_per_cell = fe->n_dofs_per_cell();
		const unsigned int n_q_points = quadrature_formula.size();
	
		// istantiate local matrix and vectors
		const FEValuesExtractors::Vector velocities(0);
		const FEValuesExtractors::Scalar pressure(dim);
		FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
		Vector<double> local_rhs(dofs_per_cell);
	
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
			fe_values.reinit(cell);
	
			local_matrix = 0;
			local_rhs = 0;
	
			// We need to know the values and the gradient of velocity on quadrature nodes (explained by Bucelli)
			fe_values[velocities].get_function_values(evaluation_point, present_velocity_values);
			fe_values[velocities].get_function_gradients(evaluation_point, present_velocity_gradients);
			fe_values[pressure].get_function_values(evaluation_point, present_pressure_values);
	
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
													+ gamma * div_phi_u[i] * div_phi_u[j] + phi_p[i] * phi_p[j])
													* fe_values.JxW(q);
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
			cell->get_dof_indices(local_dof_indices);
			
			// this object here holds a list on constraint based on the fact wheter
			// this is the initial step or not.
			const AffineConstraints<double> &constraints_used = initial_step ? nonzero_constraints : zero_constraints;
			if (assemble_matrix) {
				constraints_used.distribute_local_to_global(local_matrix, local_rhs, local_dof_indices, system_matrix, system_rhs);
			} else {
				constraints_used.distribute_local_to_global(local_rhs, local_dof_indices, system_rhs);
			}
		}
		if (assemble_matrix) {
			pressure_mass.reinit(sparsity_pattern.block(1, 1));
			pressure_mass.copy_from(system_matrix.block(1, 1));
	
			// bottom right block of the system block has to be zero
			system_matrix.block(1, 1) = 0;
		}
	}
	
	template <int dim>
	void StationaryNavierStokes<dim>::assemble_system(const bool initial_step)
	{
		assemble(initial_step, true);
	}
	
	template <int dim>
	void StationaryNavierStokes<dim>::assemble_rhs(const bool initial_step)
	{
		assemble(initial_step, false);
	}
	
	template <int dim>
	void StationaryNavierStokes<dim>::solve(const bool initial_step)
	{
		// as before we define contraints bsased on the iteration we're on
		const AffineConstraints<double> &constraints_used = initial_step ? nonzero_constraints : zero_constraints;
	
		// initialize object for solving the system
		SolverControl solver_control(system_matrix.m(), 1e-4 * system_rhs.l2_norm(), true);
		SolverFGMRES<BlockVector<double>> gmres(solver_control);
		
		// initialize ILU preconditioner with the pressure mass matrix we derived in the assemble() function
		SparseILU<double> pmass_preconditioner;
		pmass_preconditioner.initialize(pressure_mass, SparseILU<double>::AdditionalData());
	
		// initialize BlockShurPreconditioner passing the previously computed pmass precondtioner;
		const BlockSchurPreconditioner<SparseILU<double>> preconditioner(gamma, viscosity, system_matrix, pressure_mass, pmass_preconditioner);
		
		// solve using the Shur Preconditioner
		gmres.solve(system_matrix, newton_update, system_rhs, preconditioner);
		std::cout << "FGMRES steps: " << solver_control.last_step() << std::endl;
		constraints_used.distribute(newton_update);
	}
	
	/** @brief Function identifies area where the error is larger and refines the mesh
	 *  in order to create new cells there
	 */
	template <int dim>
	void StationaryNavierStokes<dim>::refine_mesh()
	{
		// error estimation
		Vector<float> estimated_error_per_cell(mesh.n_active_cells());
		const FEValuesExtractors::Vector velocity(0);
		KellyErrorEstimator<dim>::estimate(dof_handler, QGauss<dim - 1>(degree + 1), std::map<types::boundary_id, const Function<dim> *>(), present_solution, estimated_error_per_cell, fe->component_mask(velocity));
	
		// here it takes the 0.3 (30%) of the cells with the highest error from the mesh for refinement
		GridRefinement::refine_and_coarsen_fixed_number(mesh, estimated_error_per_cell, 0.3, 0.0);
		mesh.prepare_coarsening_and_refinement();

		Vector<double> flat_present_solution(dof_handler.n_dofs());

		// Copy block-wise data into the flat vector
		long unsigned int index = 0;
		for (size_t b = 0; b < present_solution.n_blocks(); ++b) {
			for (size_t i = 0; i < present_solution.block(b).size(); ++i) {
				// store the block b into the flat present solution vector
				flat_present_solution(index++) = present_solution.block(b)(i);
			}
   	    }
	
		// creating new points means it is necessary to add more dof.
		// This also means we need to store somewhere the solutiuon computed so far
		SolutionTransfer<dim, Vector<double>> solution_transfer(dof_handler);
		solution_transfer.prepare_for_coarsening_and_refinement(flat_present_solution);
		mesh.execute_coarsening_and_refinement();
	
		// redefines dof
		setup_dofs();
		// resize matrices and present solution
		initialize_system();
	
		// interpolating on the new mesh
		Vector<double> flat_tmp(dof_handler.n_dofs());
		solution_transfer.interpolate(flat_tmp);
		nonzero_constraints.distribute(flat_tmp);
		index = 0;
		for (unsigned int b = 0; b < present_solution.n_blocks(); ++b) {
			for (unsigned int i = 0; i < present_solution.block(b).size(); ++i) {
				// save the solution back to the present solution 
				present_solution.block(b)(i) = flat_tmp(index++);
			}
    	}
	}
	
	/** @brief We use the Newton method in order to solve a nonlinear system like
	 * 	this one 
	 */
	template <int dim>
	void StationaryNavierStokes<dim>::newton_iteration(const double tolerance, const unsigned int max_n_line_searches, const unsigned int max_n_refinements, const bool is_initial_step, const bool output_result)
	{
		bool first_step = is_initial_step;
		for (unsigned int refinement_n = 0; refinement_n < max_n_refinements + 1; ++refinement_n) {
			unsigned int line_search_n = 0;
			double last_res = 1.0;
			double current_res = 1.0;
			std::cout << "grid refinements: " << refinement_n << std::endl << "viscosity: " << viscosity << std::endl;
	
			// main loop here that controls the nonlinear solver given a tolerance
			while ((first_step || (current_res > tolerance)) && line_search_n < max_n_line_searches) {
				if (first_step) {
					// initialize and assemble the system in the first iter
					setup_dofs();
					initialize_system();
					evaluation_point = present_solution;
					assemble_system(first_step);
	
					solve(first_step);
					present_solution = newton_update;
					nonzero_constraints.distribute(present_solution);
					first_step = false;
					evaluation_point = present_solution;
					assemble_rhs(first_step);
					current_res = system_rhs.l2_norm();
					std::cout << "The residual of initial guess is " << current_res << std::endl;
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
						nonzero_constraints.distribute(evaluation_point);
						assemble_rhs(first_step);
						current_res = system_rhs.l2_norm();
						std::cout << "  alpha: " << std::setw(10) << alpha << std::setw(0) << "  residual: " << current_res << std::endl;
						if (current_res < last_res)
							break;
					}
					{
						present_solution = evaluation_point;
						std::cout << "  number of line searches: " << line_search_n << "  residual: " << current_res << std::endl;
						last_res = current_res;
					}
					++line_search_n; // increment line search number
				}
				if (output_result) {
					output_results(max_n_line_searches * refinement_n + line_search_n);
					if (current_res <= tolerance)
						process_solution(refinement_n);
				}
			}
			if (refinement_n < max_n_refinements) {
				refine_mesh();
			}
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
			std::cout << "Searching for initial guess with Re = " << Re << std::endl;
			newton_iteration(1e-12, 50, 0, is_initial_step, false);
			is_initial_step = false;
		}
	}
	
	
	/** @brief Function that just prints the output (pretty similar to Bucelli's) 
	*/
	template <int dim>
	void StationaryNavierStokes<dim>::output_results(const unsigned int output_index) const
	{
		// vector of strings with dim element as "velocity" and last element "pressure"
		std::vector<std::string> solution_names(dim, "velocity");
		solution_names.emplace_back("pressure");
	
		std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
		data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
	
		DataOut<dim> data_out;
		data_out.attach_dof_handler(dof_handler);
		data_out.add_data_vector(present_solution, solution_names, DataOut<dim>::type_dof_data, data_component_interpretation);
		data_out.build_patches();
	
		const std::string output_file_name = std::to_string(1.0 / viscosity) + "-solution";
		std::ofstream output(output_file_name + Utilities::int_to_string(output_index, 4) + ".vtk");
		data_out.write_vtk(output);
	
		std::cout << "Output written to " << output_file_name << "." << std::endl;
	}
	
	/** @brief Function that it samples the velocity profile along the vertical centerline
	 *  of the domain and writes it to a text file. Usefull for testing and validation purposes.
	 */
	template <int dim>
	void StationaryNavierStokes<dim>::process_solution(unsigned int refinement)
	{
		std::ofstream f(std::to_string(1.0 / viscosity) + "-line-" + std::to_string(refinement) + ".txt");
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
	void StationaryNavierStokes<dim>::run(const unsigned int refinement)
	{
		GridGenerator::hyper_cube(mesh);
		mesh.refine_global(5);
		const double Re = 1.0 / viscosity;
		if (Re > 1000.0) {
			std::cout << "Searching for initial guess ..." << std::endl;
			const double step_size = 2000.0;
			compute_initial_guess(step_size);
			std::cout << "Found initial guess." << std::endl;
			std::cout << "Computing solution with target Re = " << Re << std::endl;
			viscosity = 1.0 / Re;
			newton_iteration(1e-12, 50, refinement, false, true);
		} else {
			newton_iteration(1e-12, 50, refinement, true, true);
		}
	}
	
	// Explicit instantiation for dim=2
	template class StationaryNavierStokes<2>;	
};// namespace NavierStokes
