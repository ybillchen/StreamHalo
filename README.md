# StreamHalo

A Python package for generating mock stellar halos composed of tidal streams from satellite galaxies.

## Overview

StreamHalo uses particle spray methods and StreaMax to simulate stellar streams in mock galaxy halos. The package:

- Generates satellite galaxy populations following mass functions
- Creates tidal streams from satellite progenitors using StreaMax (JAX-accelerated)
- Assembles complete mock stellar halos with background populations

Leverages StreaMax's 10 built-in potential models and JAX acceleration for fast stream generation.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from streamhalo.halo import MockHalo

# Define host potential parameters (StreaMax NFW potential)
host_potential = 'NFW'
host_params = {
    'logM': 12.0,           # Log10(mass in solar masses)
    'Rs': 25.0,             # Scale radius in kpc
    'a': 1.0, 'b': 1.0, 'c': 1.0,  # Triaxial flattening
}

# Create mock halo with 10 satellites
halo = MockHalo(
    host_potential_type=host_potential,
    host_params=host_params,
    n_satellites=10,
    n_stream_particles=1000
)

# Generate individual streams
import numpy as np
time = np.linspace(0, 10, 100)  # 10 Gyr integration

satellite_ic = np.array([100, 0, 0, 50, 50, 50])  # Initial conditions
satellite_params = {'logM': 8.0, 'Rs': 1.0}

stream = halo.add_stream(
    satellite_ic,
    satellite_potential_type='Plummer',
    satellite_params=satellite_params,
    time=time
)
```

## Available Potentials

StreaMax supports 10 potential models:

1. **PointMass** - Simple inverse-distance
2. **Isochrone** - Core-softened spherical
3. **Plummer** - Softened point mass
4. **NFW** - Navarro-Frenk-White (triaxial)
5. **MiyamotoNagai** - Realistic disk galaxy
6. **Hernquist** - Elliptical galaxy profile
7. **Logarithmic** - Isothermal potential
8. **ExpDisk** - Exponential disk
9. **Bar** - Time-dependent bar
10. **NFW_MiyamotoNagai** - Composite NFW + disk

## Package Structure

- `satellites.py` - Satellite galaxy generation with mass functions
- `streams.py` - Stream generation using StreaMax particle spray methods
- `halo.py` - Main mock halo orchestration
- `utils.py` - Utility functions (I/O, visualization, analysis)

## Dependencies

- numpy, scipy, astropy - Scientific computing
- StreaMax - JAX-accelerated particle spray stream simulator

## Satellite Mass Function

StreamHalo uses power-law mass functions for satellite galaxies:

```python
from streamhalo.satellites import SatellitePopulation

# Create population with power-law mass function (α = -1.9)
sat_pop = SatellitePopulation(
    mass_function='powerlaw',
    alpha=-1.9,          # Typical value for dwarf galaxies
    M_min=1e7,           # 10^7 solar masses
    M_max=1e10,          # 10^10 solar masses
)

# Sample satellites with orbital parameters
satellites = sat_pop.generate(
    n_satellites=10,
    orbital_parameters={
        'velocity_scale': 100.0,      # km/s
        'radial_range': (20.0, 150.0) # kpc
    }
)

# Access StreaMax-ready data
logM = satellites['logM']              # log10 masses
initial_conditions = satellites['initial_conditions']  # [x,y,z,vx,vy,vz]
```

## TODO

- [x] Implement mass function sampling for satellite generation
- [ ] Complete halo assembly pipeline
- [ ] Add background stellar population generation
- [ ] Add I/O utilities for saving/loading halos
- [ ] Add visualization functions
- [ ] Add unit tests
- [ ] Add example notebooks with different potentials
