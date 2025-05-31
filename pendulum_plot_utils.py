# pendulum_plot_utils.py

from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.text

class PendulumPlotInitializer:
    """
    Manages the creation and initial styling of Matplotlib figures, axes,
    and all static and animated artists for the pendulum visualization.
    """
    
    # Constants for pendulum size and styling
    length_pend: float = 1.0
    length_rect_long: float = 0.3
    length_rect_short: float = 0.2
    
    # Color palette for different simulation runs
    colors: List[np.ndarray] = [
        np.array([47/255, 82/255, 143/255]),  # A shade of blue
        np.array([200/255, 50/255, 50/255]), # A shade of red
        np.array([50/255, 150/255, 50/255]), # A shade of green
        np.array([255/255, 165/255, 0/255]) # Orange
    ]
    reference_color: str = 'gray'
    alpha_value: float = 0.7 # Default transparency

    def __init__(self):
        """
        Initializes the plot initializer with default visual constants.
        These could be made configurable via parameters if needed.
        """
        pass # No specific initialization needed for now, constants are class attributes

    def create_figure_and_axes(self) -> Tuple[Figure, Axes, Axes]:
        """
        Creates the main figure and two subplots (time and pendulum animation).
        """
        fig, (ax_time, ax_pend) = plt.subplots(1, 2, figsize=(12, 6))
        return fig, ax_time, ax_pend

    def setup_time_axis(self, ax_time: Axes, min_time: float, max_time: float, 
                        min_angle_deg: float, max_angle_deg: float) -> None:
        """
        Configures the properties of the angle vs. time subplot.
        """
        ax_time.set_title("Angle over Time")
        ax_time.set_xlabel("Time / s")
        ax_time.set_ylabel("Angle / deg")
        ax_time.set_xlim(min_time * 0.8, max_time * 1.05)
        ax_time.set_ylim(min_angle_deg * 1.2, max_angle_deg * 1.2)
        ax_time.grid(True)
        ax_time.legend()

    def setup_pendulum_axis(self, ax_pend: Axes) -> None:
        """
        Configures the properties of the pendulum animation subplot.
        """
        ax_pend.set_xlim(-self.length_pend * 1.2, self.length_pend * 1.2)
        ax_pend.set_ylim(-self.length_pend * 1.2, self.length_pend * 1.2)
        ax_pend.set_aspect('equal')
        ax_pend.set_xlabel('')
        ax_pend.set_ylabel('')
        ax_pend.grid(True, which="both")
        ax_pend.tick_params(
            axis="both", which="both", length=0, labelbottom=False, labelleft=False
        )
        ax_pend.set_title("Pendulum Animation")

    def create_time_plot_artists(self, ax_time: Axes, name: str, color: np.ndarray) -> Tuple[plt.Line2D, plt.Line2D]:
        """
        Creates and returns the line and marker artists for a single simulation run's
        time plot.
        """
        line, = ax_time.plot([], [], lw=1, color=color, label=f'Integrator ({name})', alpha=self.alpha_value)
        marker, = ax_time.plot([], [], 'o', ms=8, color=color, alpha=self.alpha_value) # marker initial data is set in _init_animation
        return line, marker

    def create_pendulum_artists(self, ax_pend: Axes, color: np.ndarray) -> Tuple[plt.Line2D, Rectangle]:
        """
        Creates and returns the line and rectangle artists for a single simulation run's
        pendulum animation.
        """
        line, = ax_pend.plot([], [], lw=2, color=color, linestyle='-', zorder=2, alpha=self.alpha_value)
        rect = Rectangle(
            (-self.length_rect_long/2, -self.length_rect_short/2),
            self.length_rect_long,
            self.length_rect_short,
            facecolor=color,
            edgecolor='black',
            linewidth=2,
            transform=ax_pend.transData, # Use ax_pend.transData directly
            zorder=1,
            alpha=self.alpha_value
        )
        ax_pend.add_patch(rect)
        return line, rect

    def create_reference_time_line(self, ax_time: Axes) -> plt.Line2D:
        """
        Creates and returns the line artist for the reference solution on the time plot.
        """
        line, = ax_time.plot([], [], color=self.reference_color, linestyle='--', label='Reference (RK45)', alpha=self.alpha_value)
        return line

    def create_reference_pendulum_artists(self, ax_pend: Axes) -> Tuple[plt.Line2D, Rectangle]:
        """
        Creates and returns the line and rectangle artists for the reference pendulum animation.
        """
        ref_line, = ax_pend.plot([], [], lw=2, color=self.reference_color, linestyle='--', zorder=0, alpha=self.alpha_value)
        ref_rect = Rectangle(
            (-self.length_rect_long/2, -self.length_rect_short/2),
            self.length_rect_long,
            self.length_rect_short,
            facecolor=self.reference_color,
            edgecolor='black',
            linewidth=2,
            transform=ax_pend.transData, # Use ax_pend.transData directly
            zorder=0,
            alpha=self.alpha_value
        )
        ax_pend.add_patch(ref_rect)
        return ref_line, ref_rect

    def create_time_text_artist(self, ax_pend: Axes) -> matplotlib.text.Text:
        """
        Creates and returns the text artist for displaying time in the animation.
        """
        return ax_pend.text(
            0.5, 0.95,
            "",
            transform=ax_pend.transAxes,
            ha="center",
            va="top",
            fontsize=12
        )
    
    def create_pivot_point_artist(self, ax_pend: Axes) -> None:
        """
        Creates the static pivot point artist. This does not need to be returned
        as it's not animated or updated per frame.
        """
        ax_pend.scatter(0, 0, s=200, c='black', marker='o', zorder=10, alpha=self.alpha_value)