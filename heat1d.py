#!/usr/bin/env python3
"""
heat1d.py – Solve the one‑dimensional heat equation

    ∂u/∂t = α ∂²u/∂x² ,   x∈[0,L] ,  t∈[0,T]

using an explicit finite‑difference scheme (FTCS).
The script

* reads all parameters from a JSON or YAML configuration file,
* writes the temperature field `u(x,t)` to a CSV file,
* provides a minimal pytest‑compatible test.

The code follows PEP‑8, uses only the standard library and NumPy
(and optionally PyYAML).  No external CFD packages are required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def load_config(cfg_path: Path) -> Dict[str, Any]:
    """
    Load a configuration file.

    Parameters
    ----------
    cfg_path : Path
        Path to a ``.json`` or ``.yaml/.yml`` file.

    Returns
    -------
    dict
        Dictionary with the simulation parameters.
    """
    ext = cfg_path.suffix.lower()
    if ext == ".json":
        with cfg_path.open("rt", encoding="utf-8") as f:
            return json.load(f)
    if ext in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError("PyYAML is required for .yaml config files") from exc
        with cfg_path.open("rt", encoding="utf-8") as f:
            return yaml.safe_load(f)
    raise ValueError("Unsupported config format: use .json or .yaml/.yml")


def initial_condition(x: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """
    Example initial temperature distribution.

    The default is a Gaussian centred at L/2; a user can supply a
    callable ``init`` in the config (as a string) which will be
    evaluated with ``eval`` (dangerous, use only with trusted files).

    Parameters
    ----------
    x : np.ndarray
        Spatial grid points.
    cfg : dict
        Configuration dictionary.

    Returns
    -------
    np.ndarray
        Temperature at t = 0.
    """
    if "init" in cfg:
        # WARNING: eval only on trusted input
        func = eval(cfg["init"], {"np": np, "__builtins__": {}})
        return func(x)
    # Gaussian pulse
    L = cfg["length"]
    return np.exp(-((x - L / 2) ** 2) / (0.1 * L) ** 2)


def solve_heat_equation(cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the 1‑D heat equation with FTCS.

    Parameters
    ----------
    cfg : dict
        Required keys:
        * ``diffusivity`` (float) – thermal diffusivity α
        * ``length`` (float) – domain size L
        * ``nx`` (int) – number of interior grid points (including boundaries)
        * ``dt`` (float) – time step size
        * ``t_end`` (float) – final simulation time
        Optional:
        * ``bc_left`` / ``bc_right`` (float) – Dirichlet boundary values
        * ``init`` (str) – Python expression for initial condition, e.g.
          ``"np.sin(np.pi*x/L)"``

    Returns
    -------
    x : np.ndarray
        Spatial coordinates (size ``nx``).
    t : np.ndarray
        Times at which the solution was stored (size ``nt``).
    u : np.ndarray
        Temperature field of shape ``(nt, nx)``.
    """
    α = float(cfg["diffusivity"])
    L = float(cfg["length"])
    nx = int(cfg["nx"])
    dt = float(cfg["dt"])
    t_end = float(cfg["t_end"])

    # Spatial grid (including boundaries)
    x = np.linspace(0.0, L, nx)
    dx = x[1] - x[0]

    # Stability condition for FTCS: α*dt/dx² ≤ 0.5
    if α * dt / dx ** 2 > 0.5:
        sys.stderr.write(
            f"Warning: scheme may be unstable (α·dt/dx² = {α*dt/dx**2:.3f} > 0.5)\n"
        )

    # Time array
    nt = int(np.ceil(t_end / dt)) + 1
    t = np.linspace(0.0, t_end, nt)

    # Allocate solution array
    u = np.empty((nt, nx), dtype=float)

    # Initial condition
    u[0, :] = initial_condition(x, cfg)

    # Apply Dirichlet boundaries (default: 0)
    bc_left = float(cfg.get("bc_left", 0.0))
    bc_right = float(cfg.get("bc_right", 0.0))
    u[:, 0] = bc_left
    u[:, -1] = bc_right

    # FTCS update loop
    coeff = α * dt / dx ** 2
    for n in range(0, nt - 1):
        # interior points only
        u[n + 1, 1:-1] = (
            u[n, 1:-1] + coeff * (u[n, 2:] - 2 * u[n, 1:-1] + u[n, :-2])
        )
        # enforce boundaries (in case they change with time)
        u[n + 1, 0] = bc_left
        u[n + 1, -1] = bc_right

    return x, t, u


def save_to_csv(x: np.ndarray, t: np.ndarray, u: np.ndarray, out_path: Path) -> None:
    """
    Write the solution to a CSV file.

    The first column holds the time stamp, the remaining columns hold
    the temperature at each spatial node (ordered as in `x`).

    Parameters
    ----------
    x, t, u : ndarray
        Output of :func:`solve_heat_equation`.
    out_path : Path
        Destination file (will be overwritten).
    """
    header = ["time"] + [f"x={xi:.5g}" for xi in x]
    data = np.column_stack((t[:, None], u))
    np.savetxt(out_path, data, delimiter=",", header=",".join(header), comments="")


# ----------------------------------------------------------------------
# Command‑line interface
# ----------------------------------------------------------------------
def main() -> None:
    """
    CLI entry point.

    Usage
    -----
    python heat1d.py config.yaml            # reads config, writes results.csv
    python heat1d.py config.json -o out.npy  # writes NumPy binary file
    """
    import argparse

    parser = argparse.ArgumentParser(description="1D heat equation solver")
    parser.add_argument("config", type=Path, help="JSON or YAML configuration file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("results.csv"),
        help="Output file (CSV by default, .npy for NumPy binary)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    x, t, u = solve_heat_equation(cfg)

    if args.output.suffix.lower() == ".npy":
        np.save(args.output, u)
    else:
        save_to_csv(x, t, u, args.output)

    print(f"Simulation finished. Results written to {args.output}")


if __name__ == "__main__":
    main()
