# StreaMax Potential Models Reference

StreamHalo inherits all 10 galactic potential models from StreaMax. This guide details the required and optional parameters for each.

## Common Parameters

All potentials support position offset:
- `x_origin`, `y_origin`, `z_origin` - Center position (default: 0, 0, 0)

## Host Potentials

### 1. PointMass
Simplest potential: Phi = -GM/r

**Parameters:**
- `M` or `logM` - Mass (direct or log10 solar masses)

**Example:**
```python
{'logM': 11.0, 'x_origin': 0, 'y_origin': 0, 'z_origin': 0}
```

---

### 2. Isochrone
Core-softened spherical potential: Phi = -GM/(b + sqrt(r^2 + b^2))

**Parameters:**
- `M` or `logM` - Mass
- `Rs` - Core/softening radius (kpc)

**Example:**
```python
{'logM': 12.0, 'Rs': 2.0}
```

---

### 3. Plummer
Softened point mass: Phi = -GM/sqrt(r^2 + Rs^2)

**Parameters:**
- `M` or `logM` - Mass
- `Rs` - Softening radius (kpc)

**Example:**
```python
{'logM': 8.0, 'Rs': 1.0}
```

---

### 4. NFW
Navarro-Frenk-White dark matter profile with triaxial support

**Parameters:**
- `M` or `logM` - Mass
- `Rs` - Scale radius (kpc)
- `a`, `b`, `c` - Triaxial axis ratios (default: 1, 1, 1 for spherical)
- `dirx`, `diry`, `dirz` - Direction vector for major axis (optional)

**Example (spherical):**
```python
{
    'logM': 12.0,
    'Rs': 25.0,
    'a': 1.0, 'b': 1.0, 'c': 1.0,
}
```

**Example (triaxial):**
```python
{
    'logM': 12.0,
    'Rs': 25.0,
    'a': 1.2, 'b': 1.1, 'c': 1.0,  # Flattened along c-axis
    'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
}
```

---

### 5. MiyamotoNagai
Realistic disk galaxy model combining bulge and disk components

**Parameters:**
- `M` or `logM` - Mass
- `Rs` - Disk scale radius (kpc)
- `Hs` - Vertical scale height (kpc)

**Example:**
```python
{
    'logM': 11.5,
    'Rs': 4.0,   # Disk scale radius
    'Hs': 0.3,   # Vertical scale
}
```

---

### 6. Hernquist
Elliptical galaxy profile: Phi = -GM/(r + Rs)

**Parameters:**
- `M` or `logM` - Mass
- `Rs` - Scale radius (kpc)

**Example:**
```python
{'logM': 11.0, 'Rs': 5.0}
```

---

### 7. Logarithmic
Isothermal potential with triaxial symmetry: Phi = 0.5*V0^2*log(...)

**Parameters:**
- `V0` - Central velocity dispersion (km/s)
- `Rc` - Core radius (kpc)
- `q1`, `q2` - Axis ratios for triaxiality
- `dirx`, `diry`, `dirz` - Direction vector (optional)

**Example:**
```python
{
    'V0': 200.0,  # km/s
    'Rc': 10.0,
    'q1': 0.9,
    'q2': 0.8,
}
```

---

### 8. ExpDisk
Exponential disk using modified Bessel functions (I0, K1)

**Parameters:**
- `Sigma0` or `logSigma0` - Surface density (direct or log10)
- `Rs` - Disk scale radius (kpc)
- `Hs` - Vertical scale height (kpc)

**Example:**
```python
{
    'logSigma0': 8.5,
    'Rs': 3.0,
    'Hs': 0.25,
}
```

---

### 9. Bar
Time-dependent quadratic bar potential with Gaussian vertical profile

**Parameters:**
- `A` - Bar amplitude
- `Rs` - Scale radius (kpc)
- `Hs` - Vertical scale (kpc)
- `Omega` - Bar rotation frequency (1/Gyr)
- `t0` - Bar activation time (Gyr)
- `t1` - Bar deactivation time (Gyr)

**Example:**
```python
{
    'A': 0.01,
    'Rs': 3.0,
    'Hs': 0.3,
    'Omega': 1.5,  # 1/Gyr
    't0': 2.0,     # Activate at 2 Gyr
    't1': 8.0,     # Deactivate at 8 Gyr
}
```

---

### 10. NFW_MiyamotoNagai
Composite potential: NFW halo + MiyamotoNagai disk

**Parameters:**
Pass parameters as nested dictionaries for each component:

**Example:**
```python
{
    # NFW halo component
    'nfw': {
        'logM': 12.0,
        'Rs': 25.0,
        'a': 1.0, 'b': 1.0, 'c': 1.0,
    },
    # MiyamotoNagai disk component
    'mn': {
        'logM': 10.5,
        'Rs': 4.0,
        'Hs': 0.3,
    }
}
```

---

## Satellite Potentials

Satellites typically use simpler potentials:
- **Plummer** - Default choice
- **Isochrone** - Spherical alternative
- **PointMass** - Simplest (no softening)
- **NFW** - For massive subhalos

---

## Tips for Choosing Potentials

1. **Default Setup:** NFW host + Plummer satellites
2. **Disk Galaxy:** NFW_MiyamotoNagai host (realistic Milky Way-like)
3. **Spherical Systems:** Hernquist or Isochrone
4. **Isolated Disks:** ExpDisk or MiyamotoNagai
5. **Time-Varying:** Bar potential for perturbations

---

## Unit Conventions

All positions and velocities are in:
- **Position:** kpc
- **Velocity:** km/s
- **Mass:** Solar masses (M_sun)
- **Time:** Gyr
- **Frequency:** 1/Gyr (for rotation rates)
