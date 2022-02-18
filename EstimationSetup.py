

from kernel.numerical_simulation import environment_setup

from kernel.numerical_simulation.estimation_setup import observation
from kernel import constants
from kernel.astro import element_conversion

import numpy as np


def setup_ground_stations(bodies):
    # New Norcia
    environment_setup.add_ground_station(
        bodies.get_body('Earth'), 'Norcia', [0.0, -0.54175, 2.0278], element_conversion.geodetic_position_type)
    # Madrid
    environment_setup.add_ground_station(
        bodies.get_body('Earth'), 'Madrid', [0.0, 0.70562, -0.07416], element_conversion.geodetic_position_type)
    # Goldstone
    environment_setup.add_ground_station(
        bodies.get_body('Earth'), 'Goldstone', [0.0, 0.61831, -2.04011], element_conversion.geodetic_position_type)


def create_link_ends(bodies):

     # link_ends =  observations.one_way_uplink_link_ends([("Earth", "Goldstone"),
     # ("Earth", "Madrid"), ("Earth", "Norcia")], ("Vehicle", "") )

    ground_stations = environment_setup.get_ground_station_list(bodies.get_body( "Earth" ))
    link_ends = observation.one_way_downlink_link_ends(("Vehicle", ""), ground_stations)

    # three_way_link_ends_1 = dict();
    # three_way_link_ends_1[observation.transmitter] = ("Earth", "Goldstone");
    # three_way_link_ends_1[observation.reflector1] = ("Vehicle", "");
    # three_way_link_ends_1[observation.receiver] = ("Earth", "Goldstone");
    #
    # three_way_link_ends_2 = dict();
    # three_way_link_ends_2[observation.transmitter] = ("Earth", "Goldstone");
    # three_way_link_ends_2[observation.reflector1] = ("Vehicle", "");
    # three_way_link_ends_2[observation.receiver] = ("Earth", "Madrid");

    return link_ends

def get_observation_settings( link_ends,
                              doppler_t_int,
                              range_bias,
                              number_stations):

    range_bias_settings = observation.absolute_bias(np.array([range_bias]))
    light_time_correction_settings = [observation.first_order_relativistic_light_time_correction(['Sun'])]

    observation_settings = list()
    for i in range(number_stations):
        observation_settings.append(observation.one_way_range(link_ends[i],
            light_time_correction_settings = light_time_correction_settings,
            bias_settings = range_bias_settings ) )
        observation_settings.append(observation.one_way_closed_loop_doppler(link_ends[i],
            doppler_t_int, light_time_correction_settings = light_time_correction_settings))
        # observation_settings.append(observation.one_way_open_loop_doppler(link_ends[i]))

        #observation_settings.append(one_way_range_settings,
        #                            bias_settings = range_bias_settings)

    return observation_settings


def get_observation_simulation_settings( link_ends,
                                     range_noise,
                                     doppler_noise,
                                     range_interval,
                                     doppler_interval,
                                     initial_epoch, final_epoch,
                                     number_stations):

    buffer = 3600
    range_observation_times = np.arange( initial_epoch + buffer, final_epoch - buffer, range_interval )
    doppler_observation_times = np.arange( initial_epoch + buffer, final_epoch - buffer, doppler_interval )

    observation_simulation_settings_list = list()
    for i in range(number_stations):
        observation_simulation_settings_list.append(
            observation.tabulated_simulation_settings(
                observable_type = observation.one_way_range_type,
                link_ends = link_ends[i],
                simulation_times = range_observation_times ))
        observation_simulation_settings_list.append(
            observation.tabulated_simulation_settings(
                # observable_type=observation.one_way_doppler_type,
                observable_type=observation.one_way_differenced_range_type,
                link_ends=link_ends[i],
                simulation_times=doppler_observation_times))

    observation.add_gaussian_noise_to_settings(
        observation_simulation_settings_list,
        range_noise,
        observation.one_way_range_type )
    observation.add_gaussian_noise_to_settings(
        observation_simulation_settings_list,
        doppler_noise,
        # observation.one_way_doppler_type )
        observation.one_way_differenced_range_type )

    for i, name in zip(range(number_stations), ("Goldstone","Norcia","Madrid")):
        elevation = observation.elevation_angle_viability(("Earth", name), np.deg2rad(15))
        avoidance = observation.body_avoidance_viability(("Earth", name), "Sun", np.deg2rad(30))
        observation.add_viability_check_to_settings(
            observation_simulation_settings_list,
            [elevation, avoidance])

    return observation_simulation_settings_list


