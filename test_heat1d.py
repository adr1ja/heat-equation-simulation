import pathlib
import json
import numpy as np
from heat1d import solve_heat_equation, load_config

def test_solver_consistency(tmp_path: pathlib.Path) -> None:
    """Run a short simulation and check array shapes and stability bound."""
    cfg = {
        "diffusivity": 1.0,
        "length": 1.0,
        "nx": 21,
        "dt": 0.0005,
        "t_end": 0.01,
        "bc_left": 0.0,
        "bc_right": 0.0,
    }

    # write temporary json config (demonstrates load_config)
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    loaded = load_config(cfg_path)

    x, t, u = solve_heat_equation(loaded)

    assert x.shape == (cfg["nx"],)
    assert t.shape == (int(np.ceil(cfg["t_end"] / cfg["dt"])) + 1,)
    assert u.shape == (t.size, x.size)
    # FTCS stability check – should be ≤0.5
    dx = x[1] - x[0]
    assert cfg["diffusivity"] * cfg["dt"] / dx ** 2 <= 0.5 + 1e-12
