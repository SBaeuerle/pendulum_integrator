from typing import Optional, Tuple
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D 
from matplotlib.axes import Axes
from matplotlib.figure import Figure

matplotlib.use('TkAgg')


class VisualizePendulum():
    values_time: np.ndarray
    values_angle: np.ndarray
    ref_step_width: float = 0.01
    init_step_width: float
    step_width: float
    length_pend: float = 1.0
    length_rect_long = 0.3
    length_rect_short = 0.2
    fig: Figure
    ax_time: Axes
    ax_pend: Axes
    line_time: plt.Line2D
    marker_time: plt.Line2D
    line_pend: plt.Line2D
    text_pend:  matplotlib.text.Text
    rect: Rectangle
    ani: Optional[FuncAnimation]

    blue: np.ndarray = np.array([47/255, 82/255, 143/255])

    def __init__(self, values_time: np.ndarray, values_angle: np.ndarray) -> None:
        
        # Sample down data 
        self._assign_values(values_time, values_angle)

    def _assign_values(self, values_time: np.ndarray, values_angle: np.ndarray) -> None:
        self._assign_init_step_width(values_time)
        if self.init_step_width >= self.ref_step_width:
            self.values_time = values_time
            self.values_angle = values_angle
            self.step_width = self.init_step_width
        else:
            self._sample_data_down_and_assign(values_time, values_angle)

    def _sample_data_down_and_assign(self, values_time: np.ndarray, values_angle: np.ndarray) -> None:             
        values_time_interpolated = np.arange(values_time[0], values_time[-1] + self.ref_step_width, self.ref_step_width)
        values_angle_interpolated = np.interp(values_time_interpolated, values_time, values_angle)
        
        self.values_time = values_time_interpolated
        self.values_angle = values_angle_interpolated
        self.step_width = self.ref_step_width    

    def _assign_init_step_width(self, values_time: np.ndarray) -> None:
        self.init_step_width = values_time[1] - values_time[0]

    def _create_animation_figure(self) -> None:
        self.fig, (self.ax_time, self.ax_pend) = plt.subplots(1, 2, figsize=(12, 6))

        # --- Left: Angle vs. Time plot ---
        self.line_time, = self.ax_time.plot([], [], lw=1, color=self.blue)
        self.marker_time, = self.ax_time.plot(self.values_time[0], np.rad2deg(self.values_angle[0]), 'o', ms=8, color=self.blue)
        self.ax_time.set_title("Winkel über Zeit")
        self.ax_time.set_xlabel("Zeit / s")
        self.ax_time.set_ylabel("Winkel / grad")
        self.ax_time.set_xlim(np.min(self.values_time) * 0.8, np.max(self.values_time))
        self.ax_time.set_ylim(np.min(np.rad2deg(self.values_angle)) * 1.2, np.max(np.rad2deg(self.values_angle)) * 1.2)
        self.ax_time.grid(True)

        # --- Right: Pendulum animation ---
        self.ax_pend.set_xlim(-self.length_pend * 1.2, self.length_pend * 1.2)
        self.ax_pend.set_ylim(-self.length_pend * 1.2, self.length_pend * 1.2)
        self.ax_pend.set_aspect('equal')
        self.ax_pend.set_xlabel('')
        self.ax_pend.set_ylabel('')
        self.ax_pend.grid(True, which="both")

        # keep some ticks but hide their marks and labels
        self.ax_pend.tick_params(
            axis="both",
            which="both",       # major and minor
            length=0,           # no tick marks
            labelbottom=False,  # no x‐labels
            labelleft=False     # no y‐labels
        )

        # Initialize line and time text
        self.line_pend, = self.ax_pend.plot([], [], lw=2, color='gray')
        self.ax_pend.scatter(0, 0, s=200, c='black', marker='o')
        self.text_pend = self.ax_pend.text(
            0.5, 0.95,        # x, y in axes‐fraction coordinates
            "",               # start empty
            transform=self.ax_pend.transAxes,
            ha="center",      # horizontal alignment
            va="top",
            fontsize=12
            )

        self.rect_pend = Rectangle(
            (-self.length_rect_long/2, -self.length_rect_short/2),
            self.length_rect_long,
            self.length_rect_short,
            facecolor='white',
            edgecolor='black',
            linewidth=2,
            transform=self.ax_pend.transData)
        self.ax_pend.add_patch(self.rect_pend)
        self.ax_pend.set_title("Pendel")

    def _init_animation(self) -> Tuple:
        self.line_time.set_data([], [])
        self.marker_time.set_data([], [])
        self.text_pend.set_text("")
        self.line_pend.set_data([], [])
        self.rect_pend.set_transform(self.ax_pend.transData)
        return self.line_time, self.marker_time, self.line_pend, self.rect_pend
        
    def _update_animation(self, frame: int) -> Tuple:
        # Update time plot
        self.line_time.set_data(self.values_time[:frame], np.rad2deg(self.values_angle[:frame]))
        
        # Update time marker
        self.marker_time.set_data([self.values_time[frame]], [np.rad2deg(self.values_angle[frame])])
        
        # Data for pendulum and line
        length_string = (self.length_pend-self.length_rect_long/2)
        x: float = self.length_pend * np.sin(self.values_angle[frame])
        y: float = -self.length_pend * np.cos(self.values_angle[frame])

        # Pendulum line
        x_line: float = x*length_string/self.length_pend
        y_line: float = y*length_string/self.length_pend
        self.line_pend.set_data([0, x_line], [0, y_line])
        
        # Rotate and move rectangle bob
        angle: float = np.arctan2(y, x)
        trans = Affine2D().rotate(angle).translate(x, y) + self.ax_pend.transData
        self.rect_pend.set_transform(trans)
        self.text_pend.set_text(f"{self.values_time[frame]:.2f} s")
        return self.line_time, self.marker_time, self.line_pend, self.rect_pend

    def animate(
        self,
        repeat: bool = False
    ) -> FuncAnimation:
        
        self._create_animation_figure()
        sec_to_millisec: float = 1000

        self.ani = FuncAnimation(
            self.fig,
            self._update_animation,
            frames=len(self.values_time),
            init_func=self._init_animation,
            blit=True,
            interval=self.step_width*sec_to_millisec,
            repeat=repeat
        )
        plt.suptitle("Pendulum Animation with Time-Trace Indicator")
        plt.show()
        return self.ani

    # def plot_simulation_data(values_time, values_angle_deg, integrator_name="Euler explizit"):
    #     """
    #     Visualisiert die Ergebnisse der Pendelsimulation.
    #     (Deine bekannte Plot-Funktion)
    #     """
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(values_time, values_angle_deg, label=f"{integrator_name}")

    #     plt.title("Pendelsimulation: Winkel über Zeit")

    #     plt.xlabel("Zeit/ s")
    #     plt.ylabel("Winkel/ deg")
    #     plt.grid(True)
    #     plt.legend()
    #     plt.show()






