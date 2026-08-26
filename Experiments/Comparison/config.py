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

# Evaluation routes (Town04 spawn indices, chosen via route_survey.py for varied
# behavior): 20 highway/straight, 40 gentle curves, 0 one sharp corner, 180 twisty.
    "map": "Town04",
    "route_spawn_indices": [20, 40, 0, 180],

# ---------------------------------------------------------------------------
# Disturbances — physically grounded. All wheel parameters are expressed as
# RATIOS of the vehicle's own runtime defaults, so they stay meaningful across
# vehicles and CARLA versions. Dump the actual defaults with
#   python -m Experiments.Tuning.hpc.dump_wheel_defaults
# ---------------------------------------------------------------------------
    "steer_bias": 0.2,            # steering-offset magnitude [-1,1 command units]

# Road surface. CARLA's default tire_friction corresponds to dry asphalt.
# Scales are ratios of published tyre-road PEAK friction coefficients:
#   dry asphalt mu~0.85 | wet mu~0.50 | packed snow mu~0.25 | ice mu~0.12
    "wet_friction_scale":  0.60,
    "icy_friction_scale":  0.15,  # true ice, not packed snow (was 0.29 -> snow)

# Flat / severely under-inflated tyre on ONE wheel. A deflation is not mainly a
# peak-mu change: the dominant effects are a loss of cornering stiffness, a
# smaller rolling radius, and a large rolling-resistance rise that yields an
# asymmetric yaw moment. Modelled as scales on the corresponding wheel params.
    "flat_tire_wheel":      0,    # 0=FL, 1=FR, 2=RL, 3=RR
    "flat_lat_stiff_scale": 0.45, # cornering-stiffness collapse (dominant effect)
    "flat_radius_scale":    0.90, # deflated rolling radius
    "flat_damping_scale":   3.0,  # rolling-resistance proxy -> asymmetric drag
    "flat_friction_scale":  0.85, # peak mu falls only modestly

# Steady crosswind, applied as an aerodynamic force each step:
#   F = 0.5 * rho * V_rel^2 * A * C   (C_S laterally, C_D longitudinally)
# 80 km/h = Beaufort 9 (severe gale); at 15 m/s cruise this gives ~1.8 kN side
# force ~= 0.12 g on a 1500 kg vehicle, ~14% of available tyre grip. Severe but
# recoverable. For reference the old 15 kN wind implied a ~250 km/h relative
# wind (Category 5) and exceeded total tyre grip (mu*m*g ~ 12.5 kN) outright,
# which is why no controller could reject it.
    "wind_speed_kmh": 80.0,
    "wind_dir_deg":   90.0,       # bearing of the wind in the world XY plane
    "air_density":    1.225,      # rho [kg/m^3]
    "frontal_area":   2.2,        # A [m^2], reference area for both coefficients
    "side_force_coeff": 2.2,      # C_S at full side-on yaw
    "drag_coeff":       0.35,     # C_D

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
    "ref_points":1500,
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

