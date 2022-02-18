# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import sys

#sys.path.insert(1, '/Users/michaelplumaris/tudat-bundle/cmake-build-release/tudatpy/tudatpy')
# sys.path.insert(1, '/opt/anaconda3/envs/tudat-bundle/tudatpy/tudatpy')


import numpy as np
from kernel import constants
from kernel import numerical_simulation
from kernel.interface import spice_interface
from kernel.numerical_simulation import propagation_setup
from kernel.numerical_simulation import estimation, estimation_setup
from kernel.numerical_simulation.estimation_setup import observation
# from kernel.math import interpolators
from kernel.astro import element_conversion


from matplotlib import pyplot as plt
font_size = 20
plt.rcParams.update({'font.size': font_size})

from EstimationSetup import *
from EnvironmentSetup import *

""" TODO:
1x day bias range estimate
all stations perform uplink
add empirical stochastic to model uncertainty in re radiation
arcwise Cr"""


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    au = constants.ASTRONOMICAL_UNIT
    yr = constants.JULIAN_YEAR

    spice_interface.clear_kernels()
    spice_interface.load_standard_kernels()
    # spice_interface.load_kernel('de438.bsp')

    initial_epoch = spice_interface.convert_date_string_to_ephemeris_time('1 JUN 2025')
    final_epoch = initial_epoch + 1 * yr
    global_frame_origin = "SSB"
    global_frame_orientation = "J2000"
    bodies = get_bodies( global_frame_origin, global_frame_orientation )

    pos_jup = spice_interface.get_body_cartesian_position_at_epoch("Jupiter", "Sun", "J2000", "None", initial_epoch)

    initial_state = np.array([
        pos_jup[0] + 5e9, pos_jup[1] + 5e9, 0,
        6.8 * au / yr * np.cos(np.deg2rad(140)), 6.8 * au / yr * np.sin(np.deg2rad(140)), 0])

    # initial_state = spice_interface.get_body_cartesian_state_at_epoch(
    #     "Saturn", "Sun", "J2000", "None", initial_epoch)
    # initial_state = np.array([ 1.42598088e+12, -1.06340408e+11, -1.05324475e+11,
    #                            3.97917314e+02, 8.87843809e+03, 3.64924509e+03])

    # initial_state = element_conversion.keplerian_to_cartesian()
    bodies_to_propagate = ["Vehicle"]
    central_bodies = ["Sun"]

    acceleration_models = get_vehicle_accelerations(bodies, bodies_to_propagate, central_bodies)

    # integrator_settings = propagation_setup.integrator.bulirsch_stoer(initial_time = initial_epoch,
    #      initial_time_step = 1.0, extrapolation_sequence = propagation_setup.integrator.bulirsch_stoer_sequence,
    #       maximum_number_of_steps = 10, minimum_step_size = 1.0, maximum_step_size = np.inf)
    integrator_settings = propagation_setup.integrator.adams_bashforth_moulton(
        initial_time=initial_epoch, initial_time_step=1.0, minimum_step_size=1.0,
        maximum_step_size=np.inf, relative_error_tolerance=1.0, absolute_error_tolerance=1.0 )

    termination_condition = propagation_setup.propagator.time_termination(final_epoch)

    dependent_variables_to_save = [
        propagation_setup.dependent_variable.relative_position("Jupiter", "Sun"),
        propagation_setup.dependent_variable.relative_position("Vehicle", "Sun"),
        # propagation_setup.dependent_variable.relative_velocity("Vehicle", "Sun"),
        #propagation_setup.dependent_variable.keplerian_state("Vehicle", "Sun")
    ]
    
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies, acceleration_models, bodies_to_propagate,
        initial_state, termination_condition, output_variables= dependent_variables_to_save)

    # Add Ground Stations
    number_stations = 3
    setup_ground_stations(bodies)
    # Define Observation Links and Types
    link_ends = create_link_ends(bodies)

    # Observation Settings: Biases, Corrections, Observable types
    doppler_interval = 600
    range_interval, range_bias = 600, 1
    observation_settings = get_observation_settings(
        link_ends, doppler_interval, range_bias, number_stations )

    # Observation Simulation Settings: Noise, Viability Conditions
    doppler_noise = 3e-5 #/ constants.SPEED_OF_LIGHT # [m/s]
    range_noise = 1
    observation_simulation_settings = get_observation_simulation_settings(link_ends,
        range_noise, doppler_noise, range_interval, doppler_interval,
        initial_epoch, final_epoch, number_stations)

    # Defining the estimatable parameters
    parameter_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)
    parameter_settings.append(estimation_setup.parameter.radiation_pressure_coefficient("Vehicle"))
    # parameter_settings.append( # divide by arcs!!
    #     estimation_setup.parameter.absolute_observation_bias(link_ends, observation.one_way_range_type))
    parameter_settings.append( estimation_setup.parameter.ppn_parameter_gamma())
    # parameter_settings.append( estimation_setup.parameter.ppn_parameter_beta('Earth','Goldstone'))

    parameter_settings.append( estimation_setup.parameter.compton_wavelength())
    # parameter_settings.append( estimation_setup.parameter.equivalence_principle_lpi_violation_parameter())
    parameters_to_estimate  = estimation_setup.create_parameter_set(parameter_settings, bodies, propagator_settings)

    apriori_covariance = np.array([ 50, 50, 50, 0.01, 0.01, 0.01,
    #                                 # 0.1,
    #                                 # range_bias,
    #                                 np.inf
                                    ])**2
    inverse_apriori_covariance = np.linalg.inv( np.eye(len(apriori_covariance)) * apriori_covariance)
    # apriori_parameter_correction = np.array([ 1, 1, 1, 0.1, 0.1, 0.1])
    # Creating the Estimator object
    estimator = numerical_simulation.Estimator(
        bodies, parameters_to_estimate , observation_settings, integrator_settings, propagator_settings)

    # Perform the observations simulation
    simulated_observations = numerical_simulation.estimation.simulate_observations(
        observation_simulation_settings, estimator.observation_simulators, bodies )

    # Save the true parameters to later analyse the error
    truth_parameters = parameters_to_estimate.parameter_vector

    # Create input object for estimation, adding observations and parameter set information
    pod_input = estimation.PodInput( simulated_observations, parameters_to_estimate.parameter_set_size,
                                     # inverse_apriori_covariance = inverse_apriori_covariance,
                                     # apriori_parameter_correction = apriori_parameter_correction
                                     )
    pod_input.define_estimation_settings( reintegrate_variational_equations=False)

    # define weighting of the observations in the inversion
    weights_per_observable = \
        {estimation_setup.observation.one_way_range_type: range_noise ** -2,
         # estimation_setup.observation.one_way_doppler_type: doppler_noise ** -2
         estimation_setup.observation.one_way_differenced_range_type: doppler_noise ** -2
         }
    pod_input.set_constant_weight_per_observable(weights_per_observable)

    pod_output = estimator.perform_estimation(pod_input,
            convergence_checker = numerical_simulation.estimation.estimation_convergence_checker(1))

    observation_times = np.array(simulated_observations.concatenated_times)

    print('Number of observations: ')
    print(len(observation_times))

    print('Formal Error: ')
    print(pod_output.formal_errors)


    dynamics_simulator = numerical_simulation.SingleArcSimulator(bodies, integrator_settings, propagator_settings)
    dependent_variables = dynamics_simulator.dependent_variable_history
    dvl = np.vstack(list(dependent_variables.values()))
    time = [ ((t - initial_epoch)/ (86400*365) ) for t in list(dependent_variables.keys()) ]
    fig, ax = plt.subplots(1, 1)
    ax.plot(dvl[:, 0] / au, dvl[:, 1] / au, label="Jupiter")
    ax.plot(dvl[:, 3] / au, dvl[:, 4] / au, label="Vehicle")
    # ax.plot(time, dvl[:, 4] / au, label="Vehicle")
    ax.set_xlabel('x [AU]')
    ax.set_ylabel('y [AU]')
    ax.set_title('Trajectories in Heliocentric frame')

    print(np.linalg.norm( dvl[-1,3:6])/ au )
    
    
    # ax.plot(time, np.linalg.norm(dvl[:,6:9], axis = 1) / au * yr)
    # ax.set_ylabel('Velocity [au/yr]')
    # ax.plot(time, dvl[:,10])
    # ax.set_ylabel('Eccentricity [-]')
    # ax.set_xlabel('Time [yr]')

    ax.legend()
    plt.show()
    







# See PyCharm help at https://www.jetbrains.com/help/pycharm/
