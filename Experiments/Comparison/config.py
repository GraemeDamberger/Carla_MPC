import numpy as np
import torch.nn as nn

config = {
# Model and training
    "samples":500000,
    "batch_size": 64,
    "epochs": 1000,
    "data_path":"Data/Training_Data/training_set.npy",
    "model_path":"Data/model.pth",
    "weight_decay": 5e-4,
    "learning_rate": 0.5e-3,
    "lr_factor": 0.8,
    "lr_patience": 20,
    "scale_V": 50,

# Shared Controller
    "Np": 50,
    "N": 5,
    "M_u_lb":-np.pi/10,
    "M_u_ub":np.pi/10,
    "Q": 10,
    "R": 0,
    "kpV":100,
    "kdV": 2,
    "eps": 0.001,

# Evaluation routes (Town04 spawn indices, chosen via route_survey.py to span
# curvature AND the resulting speed profile). Surveyed at v_max=15:
#   20  kappa_mean 0.0031  kappa_max 0.021  v_mean 14.91  v_min 12.09  no hard events
#   320 kappa_mean 0.0080  kappa_max 0.121  v_mean 14.16  v_min  5.00  one moderate corner
#   0   kappa_mean 0.0132  kappa_max 0.300  v_mean 12.99  v_min  5.00  one severe corner
#   180 kappa_mean 0.0258  kappa_max 0.256  v_mean  8.84  v_min  5.00  continuously twisty
# Route 40 was replaced by 320: it had no hard events (v_min 12.04) and showed
# almost no disturbance response in v3.
    "map": "Town04",
    "route_spawn_indices": [20, 320, 0, 180],

# ---------------------------------------------------------------------------
# Disturbances — physically grounded. All wheel parameters are expressed as
# RATIOS of the vehicle's own runtime defaults, so they stay meaningful across
# vehicles and CARLA versions. Dump the actual defaults with
#   python -m Experiments.Tuning.hpc.dump_wheel_defaults
# ---------------------------------------------------------------------------
# Steering actuator bias. CARLA's steer command is normalised to
# max_steer_angle = 70 deg (measured), so 0.2 = 14 deg of road-wheel angle.
# That is a SEVERE actuator/calibration fault, not wheel misalignment (real
# alignment pull needs only 1-3 deg, i.e. steer 0.015-0.045). Kept large
# deliberately: at 0.2 the disturbance is still almost fully rejected by every
# method, so a realistic misalignment would be invisible.
    "steer_bias": 0.2,

# Road surface. CARLA's tire_friction is a PhysX tire-model multiplier, NOT a
# physical friction coefficient, and no published calibration maps the two — so
# only RATIOS transfer. Default 3.500 (measured) is taken as dry asphalt, and
# scaled by ratios of real peak tire-road friction coefficients:
#   dry asphalt mu~0.85 | wet mu~0.50 | packed snow mu~0.25 | ice mu~0.12
# Resulting CARLA values: wet 2.100, icy 0.525.
    "wet_friction_scale":  0.60,
    "icy_friction_scale":  0.15,  # true ice (v3's 1.0 absolute was 0.29 -> snow)

# Flat tire on ONE wheel. CARLA has NO tire-pressure parameter and there is no
# published model for tire deflation in CARLA (the gap was raised in CARLA
# discussion #6170 and never answered), so this is an explicitly-defined proxy
# rather than a calibrated fault. It targets the effects that dominate a
# deflation in the vehicle-dynamics literature: loss of cornering stiffness,
# reduced rolling radius, higher rolling resistance (giving an asymmetric yaw
# moment), and reduced grip as the collapsed carcass loses an effective contact
# patch. Measured defaults -> faulted values on the FL wheel:
#   tire_friction   3.500 -> 1.050 | lat_stiff_value 15.000 -> 6.750
#   radius           34.0 -> 30.6 cm | damping_rate    0.250 -> 0.750
# The friction scale reflects a reported ~70% loss of braking/cornering grip
# for a flat tire; the scale factors are chosen for plausibility, NOT calibrated,
# and no quantitative fidelity is claimed for this condition.
    "flat_tire_wheel":      0,    # 0=FL, 1=FR, 2=RL, 3=RR
    "flat_lat_stiff_scale": 0.45, # cornering-stiffness collapse
    "flat_radius_scale":    0.90, # deflated rolling radius
    "flat_damping_scale":   3.0,  # rolling-resistance proxy -> asymmetric drag
    "flat_friction_scale":  0.30, # 3.500 -> 1.050

# Steady crosswind. SIDE FORCE ONLY — CARLA already simulates longitudinal drag
# via the vehicle's own drag_coefficient (0.300 measured), so a drag term here
# would double-count it.
#   F_side = 0.5 * rho * V_rel^2 * A * C_S,   C_S = side_force_coeff * sin(beta)
# Measured on vehicle.citroen.c3 (m = 1205 kg, weight 11821 N):
#   80 km/h = Beaufort 9 (severe gale). At 15 m/s cruise the relative wind is
#   26.8 m/s at beta = 56 deg -> 1766 N = 0.149 g = 17.6% of the tire grip
#   limit (mu*m*g = 10048 N at mu=0.85). Severe but recoverable.
# For reference the legacy 15 kN wind implied a ~250 km/h relative wind and
# demanded ~150% of available grip, so no controller could reject it.
    "wind_speed_kmh": 80.0,
    "wind_dir_deg":   90.0,       # bearing of the wind in the world XY plane
    "air_density":    1.225,      # rho [kg/m^3]
    "frontal_area":   2.2,        # A [m^2], reference area for both coefficients
    "side_force_coeff": 2.2,      # C_S at full side-on yaw

# Velocity profile (curvature-aware; see simulate_carla.compute_speed_profile)
    "v_min": 5.0,
    "v_max": 15.0,            # matches the base model's training speed (was 20 →
                             #  out-of-regime tracking on fast routes)
    "a_lat_max": 3.0,             # lateral-accel budget [m/s^2]
    "a_acc_max": 2.0,             # longitudinal accel limit [m/s^2]
    "a_dec_max": 3.0,             # longitudinal decel limit [m/s^2]

# Objective normalization
    "rmse_norm_floor": 0.1,       # floor on the per-route nominal RMSE denominator [m].
                                  # Must stay below observed nominal RMSE (~0.25 m at
                                  # v_max=15) or it binds and flattens normalization.

# Plant Model
    "l": 2.5,
    "dt": 0.005,

# Online Learning (replay_buffer and residual_dynamics)
    "buffer_size": 139,#1000,
    "online_lr_replay": 4.5463954951931305e-07,#4e-7,
    "online_lr_residual": 4.5463954951931305e-07,#2e-7,
    "online_weight_decay": 0.0004272274816226789,#1e-5,

# Tube
    "K_tube": [0,0,-44.96820062851194],#[0.0, 0.0, -15.0],

# Tube Adaptive
    "K_tube_adaptive": [0,0, -39.61905791892411],#[0.0, 0.0, -15.0],
    "rbf_num_basis": 50,
    "rbf_gamma": 26.30158762851422,#80.0,
    "rbf_sigma": 0.254658253138533,#0.7,
    "rbf_weight_clip": 20.0,

# Simulation
    "sim_T": 10000,
    "ref_steps":100,
    "num_trials":1,
    "steps": 10000,
    # Reference points at 2 m spacing. A 10000-step episode (50 s) covers only
    # ~750 m at v_max=15, so 1500 points (3 km) left ~75% of the path undriven
    # and dwarfed the trajectory in plots. 500 gives 1 km — comfortable margin
    # for the MPC lookahead without the waste.
    "ref_points":500,
    "seed":26,
    "record": False,
    "no_rendering_mode": False,
    "save_plots": True,   # per-rollout diagnostic plots; tuning sets this False
}

# Disturbance conditions evaluated on every route (single source of truth for
# both the tuner and run_exp). Each is passed straight to simulate_carla.
# `surface` is None | "wet" | "icy"; `wind` enables the crosswind model.
_NO_FAULT = {"flat_tire": False, "surface": None, "wind": False}

CONDITIONS = [
    {"name": "nominal",   "steering_force": 0.0,                  **_NO_FAULT},
    {"name": "steer",     "steering_force": config["steer_bias"], **_NO_FAULT},
    {"name": "flat_tire", "steering_force": 0.0, "flat_tire": True,  "surface": None,  "wind": False},
    {"name": "icy",       "steering_force": 0.0, "flat_tire": False, "surface": "icy", "wind": False},
    {"name": "crosswind", "steering_force": 0.0, "flat_tire": False, "surface": None,  "wind": True},
]

class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 500),
            nn.Tanh(),
            nn.Linear(500, 500),
            nn.Tanh(),
            nn.Linear(500, output_size)
        )
    def forward(self, x):
        return self.net(x)

class ResidualNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 500),
            nn.Tanh(),
            nn.Linear(500, 500),
            nn.Tanh(),
            nn.Linear(500, output_size)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)

