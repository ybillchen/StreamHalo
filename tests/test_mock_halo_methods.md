# Methods: `test_mock_halo_generation` — End-to-End Pipeline Validation

## Overview

`test_mock_halo_generation` is a comprehensive integration test that exercises the full StreamHalo pipeline: host potential construction, satellite population sampling, tidal stream integration, background-particle generation, diagnostic visualization, and quantitative consistency assertions. The test is seeded with `numpy.random.default_rng(42)` for full reproducibility. All outputs are written to `tests/outputs/full_pipeline_example/`.

---

## 1. Simulation Parameters

The following global parameters govern the test run:

| Symbol | Value | Description |
|---|---|---|
| $f_\mathrm{stream}$ | 0.5 | Fraction of total particles assigned to tidal streams |
| $M_{\star,\,\mathrm{target}}$ | $10^9\,M_\odot$ | Total stellar mass target for satellite population sampling |
| $N_\mathrm{total}$ | 10,000 | Target total particle count (streams + background) |
| $N_\mathrm{stream}$ | 5,000 | Target stream particle count ($= N_\mathrm{total} \cdot f_\mathrm{stream}$) |
| $T_\mathrm{int}$ | 4.0 Gyr | Stream orbital integration time |
| $\Delta t$ | 0.5 Myr | Integration time step; gives $n_\mathrm{steps} = T_\mathrm{int}/\Delta t = 8{,}000$ |
| $N_\mathrm{min}$ | 100 | Minimum stream-particle threshold per satellite |
| $r_\mathrm{min},\,r_\mathrm{max,\,sat}$ | 2, 100 kpc | Radial bounds for satellite position sampling |
| $r_\mathrm{max,\,bg}$ | 200 kpc | Outer radial bound for background particle sampling |

---

## 2. Host Gravitational Potential

The host galaxy is represented by a four-component composite potential, intended to approximate a Milky Way-like system. All components are centred on the coordinate origin with the symmetry axis along $\hat{z}$:

| Component | Potential type | Mass ($M_\odot$) | Scale length (kpc) | Role |
|---|---|---|---|---|
| Disk | Miyamoto–Nagai | $4.77 \times 10^{10}$ | $R_s = 2.6$, $H_s = 0.3$ | Stellar disk |
| Bulge | Hernquist | $5.00 \times 10^{9}$ | $R_s = 1.0$ | Classical bulge |
| Nucleus | Hernquist | $1.81 \times 10^{9}$ | $R_s = 0.069$ | Nuclear star cluster |
| Dark halo | NFW | $5.54 \times 10^{11}$ | $R_s = 15.6$ | Dark matter halo |

The composite potential $\Phi_\mathrm{host}(\mathbf{r})$ is evaluated as the sum of all four components. The NFW component parameters are also retained separately as `nfw` for use in the tidal-radius calculation below.

---

## 3. Satellite Population

### 3a. Halo Mass Function

Satellite halo masses are drawn from a power-law mass function,

$$\frac{dN}{dM} \propto M^\alpha, \quad \alpha = -1.9,$$

truncated over the range $M \in [10^7,\,3\times10^{10}]\,M_\odot$. Sampling proceeds via inverse CDF on the cumulative power-law distribution.

### 3b. Stellar Mass Conversion

Each satellite halo mass $M_h$ is converted to a stellar mass $M_\star$ using the Behroozi et al. (2013) stellar-to-halo-mass relation (SHMR) evaluated at $z = 0$:

$$\log_{10} M_\star = \log_{10}(\varepsilon M_1) + f\!\left(\log_{10}\frac{M_h}{M_1}\right) - f(0),$$

where $M_1 = 10^{11.514}\,M_\odot$, $\varepsilon = 10^{-1.777}$, $\alpha_B = -1.412$, $\delta = 3.508$, $\gamma = 0.316$, and $f(x)$ is the Behroozi functional form (their Equation 3).

### 3c. Sampling Until Stellar Mass Target

Rather than fixing $N_\mathrm{sat}$ in advance, satellites are sampled sequentially (each from the power-law mass function, immediately converted to stellar mass via the SHMR) until their cumulative stellar mass first reaches $M_{\star,\,\mathrm{target}} = 10^9\,M_\odot$. This produces a variable but physically motivated number of satellites $N_\mathrm{sat}$.

---

## 4. Satellite Phase-Space Sampling

### 4a. Positions

Satellite galactocentric radii are drawn from a broken power-law number-density profile,

$$n(r) \propto r^{\alpha_\mathrm{inner}},\quad r < r_\mathrm{break};\qquad n(r) \propto r^{\alpha_\mathrm{outer}},\quad r \geq r_\mathrm{break},$$

with $\alpha_\mathrm{inner} = \alpha_\mathrm{outer} = -3$ (i.e., an unbroken $r^{-3}$ profile, since $r_\mathrm{break} = 100$ kpc equals $r_\mathrm{max,\,sat}$), sampled via inverse CDF over $r \in [2,\,100]\,\mathrm{kpc}$. Angular positions are drawn isotropically on the sphere: $\cos\theta \sim \mathcal{U}(-1,1)$, $\phi \sim \mathcal{U}(0, 2\pi)$.

### 4b. Velocities

For each satellite at galactocentric radius $r$, the local escape velocity $v_\mathrm{esc}(r)$ is computed from the composite host potential as

$$v_\mathrm{esc}(r) = \sqrt{2\,[\Phi_\mathrm{host}(r_\mathrm{vir}) - \Phi_\mathrm{host}(r)]},$$

with $r_\mathrm{vir} = 200$ kpc serving as the zero-point. The satellite speed is then drawn uniformly from $[0,\,v_\mathrm{esc}(r)]$, and the velocity direction is sampled isotropically (uniform $\cos\theta$ and $\phi$). This `uniform_velocity_sampler` approximates a pressure-supported, isotropic distribution without a specific dynamical model.

---

## 5. Tidal Stream Generation

### 5a. Particle Allocation

Stream particles are allocated to satellites in proportion to their stellar mass:

$$N_i = \mathrm{round}\!\left(\frac{M_{\star,i}}{\sum_j M_{\star,j}} \cdot N_\mathrm{stream}\right).$$

Satellites for which $N_i < N_\mathrm{min} = 100$ are skipped and contribute no stream particles.

### 5b. Satellite Potential

Each satellite is modelled as a Plummer sphere. Its scale radius $R_s$ is set to **10% of the King tidal radius**:

$$R_{s,i} = 0.1\,r_{t,i},\qquad r_{t,i} = r_{\mathrm{sat},i}\left(\frac{M_{h,i}}{3\,M_\mathrm{NFW}(<r_{\mathrm{sat},i})}\right)^{1/3},$$

where $M_\mathrm{NFW}(<r)$ is the NFW enclosed mass using the dark halo parameters listed in §2. This ties the satellite's internal structure to its tidal environment, making more compact satellites out of more deeply embedded ones.

### 5c. Orbit Integration

For each satellite $i$, a stream is generated by integrating test-particle orbits in the combined potential (host + satellite Plummer) for $T_\mathrm{int} = 4.0$ Gyr at $\Delta t = 0.5$ Myr ($n_\mathrm{steps} = 8{,}000$), using the `Chen2025` stream-generation method (via `StreaMAX`). The total number of integrated test particles per satellite is $n_\mathrm{gen} = n_\mathrm{steps} \times \max(1,\,\lfloor N_i / n_\mathrm{steps}\rfloor)$. If $n_\mathrm{gen} > N_i$, a random subsample of $N_i$ particles is drawn without replacement.

After all satellites are processed, stream positions and velocities are concatenated into arrays of shape $(N_\mathrm{stream,\,actual},\,3)$, and each particle is labelled with its parent satellite index.

---

## 6. Background Particle Generation

A smooth background component is generated to fill out the mock halo to the target total count. The required number of background particles is

$$N_\mathrm{bg} = \mathrm{round}\!\left(N_\mathrm{stream,\,actual} \cdot \frac{1 - f_\mathrm{stream}}{f_\mathrm{stream}}\right).$$

If $N_\mathrm{stream,\,actual} + N_\mathrm{bg} < 0.9 \cdot N_\mathrm{total}$, a warning is issued. Background positions are drawn from the same broken power-law profile used for satellites but extended to $r_\mathrm{max} = 200$ kpc; velocities are set to zero (background particles serve as a spatial prior only). Background particles receive index $-1$ to distinguish them from stream particles.

---

## 7. Diagnostic Outputs

Three figures and one NPZ cache are written after generation:

1. **Mass distribution** (`tests/outputs/full_pipeline_example/mock_halo_mass_distribution.png`): Differential number counts $dN/d\log M$ computed in 30 logarithmically spaced bins for both halo masses and stellar masses, plotted as step functions on log–log axes. Annotated with $N_\mathrm{sat}$ and $\alpha$.

2. **XY projection** (`tests/outputs/full_pipeline_example/mock_halo_xy_projection.png`): Scatter plot of all particles in the $x$–$y$ plane within $\pm 200$ kpc, with stream particles coloured by parent satellite index (`tab20` colormap) and background particles in grey.

3. **Radial number density** (`tests/outputs/full_pipeline_example/mock_halo_radial_distribution.png`): $dN/d\log r$ for stream and background particles separately, in 40 log-spaced bins over $r \in [10,\,1000]$ kpc.

4. **NPZ cache** (`tests/outputs/mock_halo_cache.npz`): Written before background generation; contains stream positions, stream velocities, stream particle–satellite index mapping, and satellite halo/stellar masses and phase-space coordinates.

---

## 8. Quantitative Assertions

The test enforces four conditions:

| Assertion | Meaning |
|---|---|
| $N_\mathrm{stream,\,actual} > 0$ | At least one satellite cleared the $N_\mathrm{min}$ threshold and produced stream particles |
| $N_\mathrm{bg,\,actual} > 0$ | Background generation succeeded |
| $\|f_\mathrm{stream,\,actual} - 0.5\| < 0.1$ | Actual stream fraction is within 10 percentage points of the target $f_\mathrm{stream} = 0.5$ |
| $r_{\log M_\star,\,N_i} > 0.4$ | Pearson correlation between $\log_{10} M_{\star,i}$ and allocated particle count $N_i$ exceeds 0.4, confirming that more-luminous satellites receive more stream particles |

The last assertion validates the mass-proportional allocation logic: a correlation below 0.4 would indicate a systematic failure in the particle-budget calculation.
