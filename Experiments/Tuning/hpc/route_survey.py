"""
Survey CARLA spawn points to pick differently-behaving routes.

For each candidate spawn index it follows the road (same logic simulate_carla
uses) and reports route length + curvature statistics, so we can pick a set of
routes that span the curvature range (straight / gentle / tight / mixed).

Run against a live CARLA server (needs CARLA_PORT), e.g. inside a short job:
    CARLA_PORT=$PORT python -m Experiments.Tuning.hpc.route_survey --step 20 --points 1500

Then set config['route_spawn_indices'] to the chosen indices.
"""

import argparse
import os

import carla
import numpy as np

from Experiments.Comparison.config import config
from Experiments.Comparison.simulate_carla import compute_speed_profile


def route_from_spawn(world, spawn_point, distance=2.0, num_points=1500):
    wp = world.get_map().get_waypoint(
        spawn_point.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    pts = []
    for _ in range(num_points):
        loc = wp.transform.location
        pts.append([loc.x, loc.y])
        nxt = wp.next(distance)
        if not nxt:
            break
        wp = nxt[0]
    return np.array(pts).T   # (2, M)


def curvature_stats(path_xy):
    P   = path_xy.T
    d   = np.diff(P, axis=0)
    seg = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
    length = float(seg.sum())
    psi  = np.arctan2(d[:, 1], d[:, 0])
    dpsi = (np.diff(psi) + np.pi) % (2 * np.pi) - np.pi
    kappa = np.abs(dpsi) / np.maximum(0.5 * (seg[:-1] + seg[1:]), 1e-6)
    # target speed the curvature profile would command (lower = twistier)
    _, v = compute_speed_profile(
        path_xy, config['a_lat_max'], config['a_acc_max'],
        config['a_dec_max'], config['v_min'], config['v_max'])
    return {
        "length_m":  length,
        "kappa_mean": float(kappa.mean()) if len(kappa) else 0.0,
        "kappa_max":  float(kappa.max()) if len(kappa) else 0.0,
        "v_mean":     float(v.mean()),
        "v_min":      float(v.min()),
        "n_points":   P.shape[0],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=20,
                    help="survey every Nth spawn point (default 20)")
    ap.add_argument("--points", type=int, default=1500,
                    help="waypoints to follow per route (default 1500)")
    ap.add_argument("--map", type=str, default=config['map'])
    args = ap.parse_args()

    port   = int(os.environ.get("CARLA_PORT", 2000))
    client = carla.Client("localhost", port)
    client.set_timeout(30.0)
    world = client.get_world()
    if args.map not in world.get_map().name:
        world = client.load_world(args.map)

    spawns = world.get_map().get_spawn_points()
    print(f"Map {args.map}: {len(spawns)} spawn points; surveying every {args.step}")
    print(f"{'idx':>5}{'length_m':>11}{'kappa_mean':>12}{'kappa_max':>11}"
          f"{'v_mean':>9}{'v_min':>8}{'n':>7}")

    rows = []
    for idx in range(0, len(spawns), args.step):
        try:
            path = route_from_spawn(world, spawns[idx], num_points=args.points)
            if path.shape[1] < 50:
                continue
            st = curvature_stats(path)
            rows.append((idx, st))
            print(f"{idx:>5}{st['length_m']:>11.1f}{st['kappa_mean']:>12.4f}"
                  f"{st['kappa_max']:>11.4f}{st['v_mean']:>9.2f}"
                  f"{st['v_min']:>8.2f}{st['n_points']:>7}")
        except Exception as e:
            print(f"{idx:>5}  skipped: {e}")

    if rows:
        rows.sort(key=lambda r: r[1]['kappa_mean'])
        print("\nSuggested spread (by mean curvature): "
              "straightest / gentle / moderate / tightest")
        picks = [rows[0], rows[len(rows)//3], rows[2*len(rows)//3], rows[-1]]
        print("route_spawn_indices =", [p[0] for p in picks])


if __name__ == "__main__":
    main()
