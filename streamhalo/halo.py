"""Mock stellar halo orchestrator: satellites → streams → background."""

import numpy as np

from .potentials import v_esc
from .satellites import SatellitePopulation
from .streams import StreamGenerator


class MockHalo:
    """Mock stellar halo with tidal streams and smooth background.

    Workflow:
        halo = MockHalo(host_potential_type, host_params, ...)
        halo.sample_satellites(stellar_mass_target=..., position_sampler=..., velocity_sampler=...)
        halo.generate_streams(target_particles=..., plummer_scale=...)
        halo.generate_background(n_particles=..., position_sampler=...)
        halo.save('cache.npz')
    """

    def __init__(self, host_potential_type, host_params, rng=None,
                 method='Chen2025',
                 mass_function='powerlaw', alpha=-1.9,
                 M_min=1e7, M_max=1e11, shmr='behroozi13'):
        self.host_potential_type = host_potential_type
        self.host_params = host_params
        self.rng = rng if rng is not None else np.random.default_rng()

        self.satellites = SatellitePopulation(
            mass_function=mass_function, alpha=alpha,
            M_min=M_min, M_max=M_max, shmr=shmr, rng=self.rng,
        )
        self.stream_gen = StreamGenerator(
            host_potential_type, host_params, method=method,
        )

        self.satellite_halo_masses = None
        self.satellite_stellar_masses = None
        self.satellite_positions = None
        self.satellite_velocities = None

        self.stream_positions = None
        self.stream_velocities = None
        self.stream_index = None
        self.particles_per_satellite = None

        self.background_positions = None
        self.background_velocities = None

    @property
    def n_satellites(self):
        if self.satellite_halo_masses is None:
            return 0
        return len(self.satellite_halo_masses)

    def sample_satellites(self, stellar_mass_target=None, n_satellites=None,
                          position_sampler=None, velocity_sampler=None):
        """Sample satellite halo masses, positions, and velocities.

        Provide exactly one of stellar_mass_target or n_satellites.
        position_sampler: rng -> (x, y, z)
        velocity_sampler: (rng, r, v_esc_r) -> (vx, vy, vz)
        """
        if (stellar_mass_target is None) == (n_satellites is None):
            raise ValueError("Provide exactly one of stellar_mass_target or n_satellites")
        if position_sampler is None or velocity_sampler is None:
            raise ValueError("Both position_sampler and velocity_sampler are required")

        if n_satellites is not None:
            halo_masses = self.satellites.sample(n_satellites)
        else:
            halo_masses = self.satellites.sample_until_stellar_mass(stellar_mass_target)

        n = len(halo_masses)
        positions = np.empty((n, 3))
        velocities = np.empty((n, 3))
        for i in range(n):
            positions[i] = position_sampler(self.rng)
            r = float(np.linalg.norm(positions[i]))
            v_e = v_esc(r, self.host_potential_type, self.host_params)
            velocities[i] = velocity_sampler(self.rng, r, v_e)

        self.satellite_halo_masses = halo_masses
        self.satellite_stellar_masses = self.satellites.stellar_mass(halo_masses)
        self.satellite_positions = positions
        self.satellite_velocities = velocities
        return self

    def generate_streams(self, target_particles, min_particles=100,
                         integration_time=4.0, n_steps=8000,
                         satellite_potential_type='Plummer', plummer_scale=1.0):
        """Generate streams for all sampled satellites with mass-proportional allocation.

        plummer_scale: float (constant Rs) or callable (r_sat, halo_mass) -> Rs.
        Satellites whose allocation falls below min_particles are skipped.
        """
        if self.satellite_halo_masses is None:
            raise RuntimeError("Must call sample_satellites() first")

        n_sats = self.n_satellites
        total_stellar = self.satellite_stellar_masses.sum()
        scale_fn = plummer_scale if callable(plummer_scale) else (lambda r, M: float(plummer_scale))

        all_pos, all_vel, all_idx = [], [], []
        per_sat = {}

        for i in range(n_sats):
            n_target = int(np.round(self.satellite_stellar_masses[i] / total_stellar * target_particles))
            if n_target < min_particles:
                continue
            per_sat[i] = n_target

            r_sat = float(np.linalg.norm(self.satellite_positions[i]))
            halo_mass = float(self.satellite_halo_masses[i])
            Rs = float(scale_fn(r_sat, halo_mass))
            n_gen = n_steps * max(1, n_target // n_steps)

            _, _, xv_stream, _ = self.stream_gen.generate_stream(
                np.hstack([self.satellite_positions[i], self.satellite_velocities[i]]),
                satellite_potential_type,
                {'logM': float(np.log10(halo_mass)), 'Rs': Rs},
                integration_time,
                n_steps=n_steps,
                unroll=False,
                n_particles=n_gen,
            )

            pos, vel = xv_stream[:, :3], xv_stream[:, 3:]
            if n_target < len(pos):
                idx = self.rng.choice(len(pos), n_target, replace=False)
                pos, vel = pos[idx], vel[idx]

            all_pos.append(pos)
            all_vel.append(vel)
            all_idx.append(np.full(len(pos), i, dtype=int))

        self.stream_positions = np.vstack(all_pos)
        self.stream_velocities = np.vstack(all_vel)
        self.stream_index = np.concatenate(all_idx)
        self.particles_per_satellite = per_sat
        return self

    def generate_background(self, n_particles, position_sampler):
        """Generate background particles via position_sampler (zero velocities)."""
        positions = np.empty((n_particles, 3))
        for i in range(n_particles):
            positions[i] = position_sampler(self.rng)
        self.background_positions = positions
        self.background_velocities = np.zeros((n_particles, 3))
        return self

    @property
    def combined_positions(self):
        parts = [self.stream_positions]
        if self.background_positions is not None:
            parts.append(self.background_positions)
        return np.vstack(parts)

    @property
    def combined_velocities(self):
        parts = [self.stream_velocities]
        if self.background_velocities is not None:
            parts.append(self.background_velocities)
        return np.vstack(parts)

    @property
    def combined_index(self):
        parts = [self.stream_index]
        if self.background_positions is not None:
            parts.append(np.full(len(self.background_positions), -1, dtype=int))
        return np.concatenate(parts)

    def save(self, path):
        """Save satellite + stream arrays to NPZ."""
        np.savez(
            path,
            stream_positions=self.stream_positions,
            stream_velocities=self.stream_velocities,
            stream_index=self.stream_index,
            satellite_halo_masses=self.satellite_halo_masses,
            satellite_stellar_masses=self.satellite_stellar_masses,
            satellite_positions=self.satellite_positions,
            satellite_velocities=self.satellite_velocities,
        )
