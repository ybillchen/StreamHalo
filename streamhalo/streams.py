"""Stream generation using StreaMax particle spray method."""

try:
    from StreaMAX.StreaMAX import generate_stream as streamax_generate_stream
except ImportError:
    raise ImportError("StreaMAX is required. Install from https://github.com/David-Chemaly/StreaMAX")


class StreamGenerator:
    """Generate stellar streams using StreaMax."""

    VALID_POTENTIALS = [
        'PointMass', 'Isochrone', 'Plummer', 'NFW', 'MiyamotoNagai',
        'Hernquist', 'Logarithmic', 'ExpDisk', 'Bar', 'NFW_MiyamotoNagai'
    ]

    def __init__(self, host_potential_type, host_params, n_particles=1000, alpha=0.5):
        """Initialize stream generator."""
        if host_potential_type not in self.VALID_POTENTIALS:
            raise ValueError(
                f"Invalid potential type '{host_potential_type}'. "
                f"Must be one of: {self.VALID_POTENTIALS}"
            )
        self.host_potential_type = host_potential_type
        self.host_params = host_params
        self.n_particles = n_particles
        self.alpha = alpha

    def generate_stream(self, satellite_initial_condition, satellite_potential_type,
                       satellite_params, time, n_steps=1000, unroll=False, n_particles=None):
        """Generate stream from satellite and return positions/velocities."""
        if satellite_potential_type not in self.VALID_POTENTIALS:
            raise ValueError(
                f"Invalid satellite potential type '{satellite_potential_type}'. "
                f"Must be one of: {self.VALID_POTENTIALS}"
            )

        # Use provided n_particles or fall back to default
        n_particles_to_use = n_particles if n_particles is not None else self.n_particles

        # StreaMAX expects n_steps to be integration steps (n_chunks - 1)
        streamax_n_steps = n_steps - 1

        t_sat, xv_sat, xv_stream, xhi_stream = streamax_generate_stream(
            satellite_initial_condition,
            self.host_potential_type,
            self.host_params,
            satellite_potential_type,
            satellite_params,
            time,
            self.alpha,
            streamax_n_steps,
            n_particles_to_use,
            unroll
        )
        return t_sat, xv_sat, xv_stream, xhi_stream
