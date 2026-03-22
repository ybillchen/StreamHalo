"""StreamHalo: Mock stellar halo generator with tidal streams."""

from .halo import MockHalo
from .satellites import SatellitePopulation
from .streams import StreamGenerator

__version__ = "0.1.0"

__all__ = ['MockHalo', 'SatellitePopulation', 'StreamGenerator']
