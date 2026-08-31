"""
Print the vehicle's default physics/wheel parameters and the values each
disturbance condition produces, plus the crosswind force at the configured
wind speed. Use this to quote real baseline numbers in the writeup instead of
assuming CARLA's documented defaults.

    CARLA_PORT=$PORT python -m Experiments.Tuning.hpc.dump_wheel_defaults
"""

import os

import carla
import numpy as np

from Experiments.Comparison.config import config

WHEEL_FIELDS = [
    "tire_friction", "damping_rate", "max_steer_angle", "radius",
    "max_brake_torque", "max_handbrake_torque",
    "lat_stiff_max_load", "lat_stiff_value", "long_stiff_value",
]
NAMES = ["FL", "FR", "RL", "RR"]


def show(wheels, title):
    print(f"\n{title}")
    print(f"  {'field':<22}" + "".join(f"{n:>12}" for n in NAMES))
    for f in WHEEL_FIELDS:
        vals = []
        for w in wheels:
            try:
                vals.append(f"{getattr(w, f):>12.3f}")
            except AttributeError:
                vals.append(f"{'n/a':>12}")
        print(f"  {f:<22}" + "".join(vals))


def main():
    port   = int(os.environ.get("CARLA_PORT", 2000))
    client = carla.Client("localhost", port)
    client.set_timeout(30.0)
    world = client.get_world()
    if config['map'] not in world.get_map().name:
        world = client.load_world(config['map'])

    bp    = world.get_blueprint_library().filter('*vehicle*')[3]
    spawn = world.get_map().get_spawn_points()[config['route_spawn_indices'][0]]
    veh   = world.spawn_actor(bp, spawn)
    world.tick()

    try:
        pc = veh.get_physics_control()
        print(f"Vehicle blueprint : {bp.id}")
        print(f"Mass              : {pc.mass:.1f} kg")
        print(f"Drag coefficient  : {pc.drag_coefficient:.3f}")
        show(pc.wheels, "DEFAULT wheel parameters")

        mg = pc.mass * 9.81
        print(f"\nWeight            : {mg:.0f} N")

        # what each surface condition produces
        for surf in ("wet", "icy"):
            s = config[f"{surf}_friction_scale"]
            print(f"\n{surf}: tire_friction x{s} -> "
                  f"{pc.wheels[0].tire_friction * s:.3f} (all wheels)")

        # flat tire
        idx = config['flat_tire_wheel']
        w   = pc.wheels[idx]
        print(f"\nflat tire on {NAMES[idx]}:")
        for field, key in [("lat_stiff_value", "flat_lat_stiff_scale"),
                           ("radius",          "flat_radius_scale"),
                           ("damping_rate",    "flat_damping_scale"),
                           ("tire_friction",   "flat_friction_scale")]:
            before = getattr(w, field)
            print(f"  {field:<18} {before:8.3f} x{config[key]:<5} -> "
                  f"{before * config[key]:8.3f}")

        # crosswind magnitude at the configured wind speed and nominal cruise
        v_w = config['wind_speed_kmh'] / 3.6
        v_c = config['v_max']
        v_rel = np.hypot(v_w, v_c)
        beta  = np.arctan2(v_w, v_c)
        q     = 0.5 * config['air_density'] * v_rel ** 2 * config['frontal_area']
        f_lat = q * config['side_force_coeff'] * np.sin(beta)
        print(f"\ncrosswind {config['wind_speed_kmh']:.0f} km/h ({v_w:.1f} m/s) "
              f"at cruise {v_c:.1f} m/s:")
        print(f"  relative wind     : {v_rel:.1f} m/s at beta = {np.rad2deg(beta):.1f} deg")
        print(f"  side force        : {f_lat:.0f} N  ({f_lat / mg:.3f} g)")
        print(f"  tire grip limit   : {0.85 * mg:.0f} N  (mu=0.85)")
        print(f"  fraction of grip  : {f_lat / (0.85 * mg) * 100:.1f} %")
    finally:
        veh.destroy()


if __name__ == "__main__":
    main()
