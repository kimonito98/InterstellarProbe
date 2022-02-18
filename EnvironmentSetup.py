



from kernel.numerical_simulation import environment_setup
from kernel import constants
from kernel.numerical_simulation import propagation_setup


def get_bodies(global_frame_origin, global_frame_orientation ):

    bodies_to_create = ["Sun", "Earth", "Jupiter"]#, "Saturn"]
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create, global_frame_origin, global_frame_orientation)

    body_settings.add_empty_settings("Vehicle")
    body_settings.get("Vehicle").constant_mass = 860

    # body_settings.get("Vehicle").ephemeris_settings = environment_setup.ephemeris.tabulated(
    #     dict(), global_frame_origin, global_frame_orientation)
    # body_settings.get("Vehicle").ephemeris_settings.make_multi_arc_ephemeris

    bodies = environment_setup.create_system_of_bodies(body_settings)

    # Radiation Pressure Interface
    reference_area_radiation = 2.5 ** 2 * constants.PI
    radiation_pressure_coefficient = 1.0
    occulting_bodies = []
    radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
        "Sun", reference_area_radiation, radiation_pressure_coefficient, occulting_bodies
    )
    environment_setup.add_radiation_pressure_interface(
        bodies, 'Vehicle', radiation_pressure_settings)


    return bodies

def get_vehicle_accelerations(bodies,
                              bodies_to_propagate,
                              central_bodies ):
    accelerations_settings_vehicle = dict(
        Sun=[
            propagation_setup.acceleration.cannonball_radiation_pressure(),
            # propagation_setup.acceleration.point_mass_gravity(),
            propagation_setup.acceleration.central_gravity_compton()

        ],
        Jupiter=[
            propagation_setup.acceleration.point_mass_gravity()
        ],
        # Saturn=[
        #     propagation_setup.acceleration.point_mass_gravity()
        # ]
    )
    acceleration_settings = {"Vehicle": accelerations_settings_vehicle}

    acceleration_models = propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings,
        bodies_to_propagate,
        central_bodies)

    acceleration_settings = {"Vehicle": accelerations_settings_vehicle}

    return propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings,
        bodies_to_propagate,
        central_bodies)






