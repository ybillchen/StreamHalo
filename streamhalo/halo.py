"""Mock stellar halo generation."""

import numpy as np
from .satellites import SatellitePopulation
from .streams import StreamGenerator


class MockHalo:
    """Mock stellar halo with tidal streams."""

    def __init__(self, host_potential_type, host_params, n_satellites=10,
                 mass_function='powerlaw', n_stream_particles=1000, rng=None):
        """Initialize halo generator."""
        self.host_potential_type = host_potential_type
        self.host_params = host_params
        self.n_satellites = n_satellites
        self.mass_function = mass_function

        self.satellites = SatellitePopulation(mass_function, rng=rng)
        self.stream_gen = StreamGenerator(
            host_potential_type, host_params, n_particles=n_stream_particles
        )
        self.halo_particles = None
        self.streams = []

    def generate(self, time, n_steps=1000, unroll=False,
                 orbital_parameters=None,
                 satellite_potential_type='Plummer',
                 satellite_scale_radius=1.0,
                 total_stream_particles=None,
                 mass_proportional=True):
        """Generate halo with streams from satellite population."""
        if satellite_potential_type not in StreamGenerator.VALID_POTENTIALS:
            raise ValueError(
                f"Invalid satellite potential type '{satellite_potential_type}'. "
                f"Must be one of: {StreamGenerator.VALID_POTENTIALS}"
            )

        sat_data = self.satellites.generate(self.n_satellites, orbital_parameters)

        if total_stream_particles is None:
            total_stream_particles = self.stream_gen.n_particles * self.n_satellites

        if mass_proportional:
            total_mass = np.sum(sat_data['masses'])
            particles_per_sat_float = (sat_data['masses'] / total_mass) * total_stream_particles
        else:
            particles_per_sat_float = np.full(self.n_satellites, total_stream_particles / self.n_satellites)

        particles_per_sat = np.round(particles_per_sat_float).astype(int)
        diff = total_stream_particles - np.sum(particles_per_sat)
        if diff != 0:
            idx_largest = np.argmax(sat_data['masses'])
            particles_per_sat[idx_largest] += diff

        divisor = n_steps
        for i in range(self.n_satellites):
            remainder = particles_per_sat[i] % divisor
            if remainder != 0:
                particles_per_sat[i] -= remainder

        new_total = np.sum(particles_per_sat)
        if new_total != total_stream_particles:
            adjustment = total_stream_particles - new_total
            particles_per_sat[np.argmax(sat_data['masses'])] += adjustment
        all_positions = []
        all_velocities = []
        all_stream_index = []

        for i in range(self.n_satellites):
            satellite_params = {
                'logM': sat_data['logM'][i],
                'Rs': satellite_scale_radius
            }

            t_sat, xv_sat, xv_stream, xhi_stream = self.add_stream(
                sat_data['initial_conditions'][i],
                satellite_potential_type,
                satellite_params,
                time,
                n_steps=n_steps,
                unroll=unroll,
                n_particles=particles_per_sat[i]
            )

            positions = xv_stream[:, :3]
            velocities = xv_stream[:, 3:]

            all_positions.append(positions)
            all_velocities.append(velocities)
            all_stream_index.append(np.full(len(positions), i, dtype=int))

        self.halo_particles = {
            'positions': np.vstack(all_positions),
            'velocities': np.vstack(all_velocities),
            'stream_index': np.concatenate(all_stream_index),
            'n_streams': self.n_satellites,
            'satellite_masses': sat_data['masses'],
        }

        return self.halo_particles

    def add_stream(self, satellite_ic, satellite_potential_type, satellite_params,
                   time, n_steps=1000, unroll=False, n_particles=None):
        """Add stream from satellite and return stream data."""
        stream_data = self.stream_gen.generate_stream(
            satellite_ic, satellite_potential_type, satellite_params,
            time, n_steps=n_steps, unroll=unroll, n_particles=n_particles
        )
        self.streams.append(stream_data)
        return stream_data

    def add_background(self, n_particles=10000, scale_radius=20.0, r_max=300.0, seed=None):
        """Add smooth background particles from Hernquist profile."""
        if self.halo_particles is None:
            raise RuntimeError("Must call generate() before add_background()")

        rng = np.random.RandomState(seed)

        # Rejection sampling for Hernquist profile: rho(r) ~ 1 / (r * (r + a)^3)
        # Sample radii with rejection method
        a = scale_radius
        positions = []

        while len(positions) < n_particles:
            # Sample candidate radii uniformly in [0, r_max]
            r_candidates = rng.uniform(0, r_max, n_particles)

            # Hernquist density profile (normalized)
            # rho(r) ~ 1 / (r * (r + a)^3)
            # Maximum density at r=0, evaluate at r_max for normalization
            rho_max = 1.0 / (scale_radius ** 4)
            rho = 1.0 / (r_candidates * (r_candidates + a) ** 3 + 1e-10)  # Avoid division by zero

            # Acceptance probability
            acceptance = rho / rho_max

            # Accept/reject
            accepted = rng.uniform(0, 1, len(r_candidates)) < acceptance
            r_accepted = r_candidates[accepted]

            # Sample positions on spheres at accepted radii
            n_accepted = len(r_accepted)
            thetas = np.arccos(rng.uniform(-1, 1, n_accepted))
            phis = rng.uniform(0, 2 * np.pi, n_accepted)

            pos = np.array([
                r_accepted * np.sin(thetas) * np.cos(phis),
                r_accepted * np.sin(thetas) * np.sin(phis),
                r_accepted * np.cos(thetas)
            ]).T

            positions.append(pos)

        # Concatenate and keep only n_particles
        bg_positions = np.vstack(positions)[:n_particles]
        bg_velocities = np.zeros((n_particles, 3))
        bg_stream_index = np.full(n_particles, -1, dtype=int)

        # Append to halo_particles
        self.halo_particles['positions'] = np.vstack([
            self.halo_particles['positions'],
            bg_positions
        ])
        self.halo_particles['velocities'] = np.vstack([
            self.halo_particles['velocities'],
            bg_velocities
        ])
        self.halo_particles['stream_index'] = np.concatenate([
            self.halo_particles['stream_index'],
            bg_stream_index
        ])
