"""Shared plotting style for StreamHalo tests.

Usage in any test file:
    from tests.plot_style import apply_style, style_ax, legend
    apply_style()   # call once at module level
"""

import os
import matplotlib.pyplot as plt

_STYLE = os.path.join(os.path.dirname(__file__), '..', 'sans.mplstyle')


def apply_style():
    """Load the shared mplstyle."""
    plt.style.use(_STYLE)


def style_ax(ax, xlabel=None, ylabel=None):
    """Apply standard axis formatting."""
    ax.tick_params(axis='both', which='both', labelsize=12)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=14)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=14)


def legend(ax, **kwargs):
    """Standard legend: no frame background, fontsize=10."""
    kwargs.setdefault('fontsize', 10)
    kwargs.setdefault('frameon', True)
    kwargs.setdefault('handlelength', 1.5)
    kwargs.setdefault('labelspacing', 0.3)
    leg = ax.legend(**kwargs)
    leg.get_frame().set_facecolor('none')
    leg.get_frame().set_edgecolor('none')
    return leg
