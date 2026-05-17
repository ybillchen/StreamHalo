# StreamHalo

Mock stellar halo generator: sample satellite galaxies, spray tidal streams, add a smooth background.

Backed by [StreaMAX](https://github.com/David-Chemaly/StreaMAX) for JAX-accelerated particle spray.

## Installation

```bash
pip install -e .
```

Requires StreaMAX installed and accessible as `StreaMAX` on the Python path.

## Quick start

```python
import numpy as np
from streamhalo import MockHalo
from streamhalo.potentials import tidal_radius
from streamhalo.sampling import make_broken_powerlaw_position_sampler, uniform_velocity_sampler

_origin = {'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
           'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0}
host_potential_type = 'Composite:MiyamotoNagai,Hernquist,Hernquist,NFW'
host_params = [
    {'logM': np.log10(4.77e10), 'Rs': 2.6,  'Hs': 0.3, **_origin},  # disk
    {'logM': np.log10(5e9),     'Rs': 1.0,             **_origin},  # bulge
    {'logM': np.log10(1.81e9),  'Rs': 0.069,           **_origin},  # nucleus
    {'logM': np.log10(5.54e11), 'Rs': 15.6,
     'a': 1.0, 'b': 1.0, 'c': 1.0,        **_origin},  # NFW halo
]
nfw = host_params[3]

halo = MockHalo(
    host_potential_type, host_params,
    mass_function='powerlaw', alpha=-1.9, M_min=1e7, M_max=3e10,
)

halo.sample_satellites(
    stellar_mass_target=1e9,
    position_sampler=make_broken_powerlaw_position_sampler(r_min=2, r_max=100),
    velocity_sampler=uniform_velocity_sampler,
)

halo.generate_streams(
    target_particles=50000,
    plummer_scale=lambda r, M: 0.1 * tidal_radius(r, M, nfw['logM'], nfw['Rs']),
)

halo.generate_background(
    n_particles=50000,
    position_sampler=make_broken_powerlaw_position_sampler(r_min=2, r_max=200),
)

halo.save('mock_halo.npz')

pos   = halo.combined_positions   # (N, 3) kpc
idx   = halo.combined_index       # satellite id, -1 = background
```

## Workflow

```
MockHalo(potential, params, mass_function=...)
    └─ sample_satellites(stellar_mass_target | n_satellites,
                         position_sampler, velocity_sampler)
    └─ generate_streams(target_particles, plummer_scale=...)
    └─ generate_background(n_particles, position_sampler)
    └─ save(path)
```

`generate_streams` allocates particles mass-proportionally across satellites and skips those below `min_particles` (default 100).

## Package structure

| Module | Contents |
|---|---|
| `halo.py` | `MockHalo` — main orchestrator |
| `satellites.py` | `SatellitePopulation` — mass function sampling + SHMR |
| `streams.py` | `StreamGenerator` — wraps StreaMAX |
| `potentials.py` | `v_esc`, `tidal_radius`, `composite_phi` |
| `sampling.py` | position/velocity sampler primitives and factories |

## Satellite mass functions

`SatellitePopulation` supports three mass functions:

| `mass_function=` | Description |
|---|---|
| `'powerlaw'` | Pure power law dN/dM ∝ M^α |
| `'truncated_powerlaw'` | Same, with exponential cutoff at `M_star` |
| `'schechter'` | Schechter function (rejection sampling) |

Stellar masses are computed via the Behroozi+2013 SHMR at z=0 (`shmr='behroozi13'`, default) or a constant fraction (`shmr='constant'`).

## Potentials

Any StreaMAX potential or composite thereof:

```python
# Single component
host_potential_type = 'NFW'
host_params = [{'logM': 12.0, 'Rs': 15.0, 'a': 1.0, 'b': 1.0, 'c': 1.0, ...}]

# Composite (colon-separated list)
host_potential_type = 'Composite:MiyamotoNagai,Hernquist,NFW'
host_params = [disk_params, bulge_params, halo_params]
```

Available components: `PointMass`, `Isochrone`, `Plummer`, `NFW`, `MiyamotoNagai`, `Hernquist`, `Logarithmic`, `ExpDisk`, `Bar`.

## Utilities

```python
from streamhalo.potentials import v_esc, tidal_radius
from streamhalo.sampling import (
    make_broken_powerlaw_position_sampler,
    make_hernquist_velocity_sampler,
    uniform_velocity_sampler,
    sample_r_broken_powerlaw,
    sample_v_hernquist,
)
```
