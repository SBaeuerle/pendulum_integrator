# visualize_pendulum.py
from typing import Optional, Tuple, List, Dict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Import the PendulumData class
from pendulum_data import PendulumData
# Import the new PendulumPlotInitializer
from pendulum_plot_utils import PendulumPlotInitializer

matplotlib.use('TkAgg') # Consider moving this to the main execution script (e.g., main.py)

class VisualizePendulum():
    # --- Attributes for visualization components ---
    fig: Figure
    ax_time: Axes
    ax_pend: Axes

    # Lists to hold matplotlib artists for each simulation run
    time_lines: List[plt.Line2D]
    time_markers: List[plt.Line2D]
    pendulum_lines: List[plt.Line2D]
    pendulum_rects: List[Rectangle]

    # Attributes for the reference pendulum animation
    reference_pendulum_data: Optional[PendulumData] = None
    reference_time_line: Optional[plt.Line2D] = None # New attribute for ref line on time plot
    reference_pendulum_line: Optional[plt.Line2D] = None
    reference_pendulum_rect: Optional[Rectangle] = None

    text_pend: matplotlib.text.Text
    ani: Optional[FuncAnimation]

    # --- Data and plotting utility ---
    pendulum_data_runs: Dict[str, PendulumData] 
    plot_initializer: PendulumPlotInitializer # New instance of the utility class

    # --- Constructor ---
    def __init__(self, simulation_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
                 reference: bool = False, m: float = 0.2, d: float = 0.2, g: float = 9.81, L: float = 0.5) -> None:
        """
        Initializes the pendulum visualization by first processing the data for multiple runs.

        Args:
            simulation_results (Dict[str, Tuple[np.ndarray, np.ndarray]]): 
                A dictionary where keys are names (e.g., 'Euler', 'RK2') and values are 
                tuples (time_values_array, state_values_array) for each simulation run.
            reference (bool): Whether to compute and show a single reference solution 
                              (applied to the first simulation run in the dictionary).
            m (float): mass of the pendulum
            d (float): Damping coefficient for the reference solution.
            g (float): inertial acceleration
            L (float): length of pendulum
        """
        # Compute the necessary parameters
        omega_0_ref: float = np.sqrt(g/L)
        D_ref: float = d/(2*m*omega_0_ref)


        self.pendulum_data_runs = {}
        self.plot_initializer = PendulumPlotInitializer() # Initialize the plotting utility

        # Get the first key to decide which run gets the reference (if 'reference' is True)
        run_names = list(simulation_results.keys())
        first_run_name = run_names[0] if run_names else None

        for name, (t_vals, u_vals) in simulation_results.items():
            run_reference_flag = reference if name == first_run_name else False
            
            current_pendulum_data = PendulumData(
                values_time=t_vals,
                values_state=u_vals,
                reference=run_reference_flag,
                omega_0_ref=omega_0_ref,
                D_ref=D_ref
            )
            self.pendulum_data_runs[name] = current_pendulum_data

            if current_pendulum_data.reference:
                self.reference_pendulum_data = current_pendulum_data

        all_step_widths = [data.step_width for data in self.pendulum_data_runs.values()]
        self.animation_step_width = min(all_step_widths)

    # --- Animation Setup --- 
    def _create_animation_figure(self) -> None:
        # Delegate figure and axes creation to the initializer
        self.fig, self.ax_time, self.ax_pend = self.plot_initializer.create_figure_and_axes()

        self.time_lines = []
        self.time_markers = []
        self.pendulum_lines = []
        self.pendulum_rects = []
        
        min_angle_deg = 0
        max_angle_deg = 0
        min_time = float('inf')
        max_time = float('-inf')

        # --- Left: Angle vs. Time plot ---
        for i, (name, data) in enumerate(self.pendulum_data_runs.items()):
            color = self.plot_initializer.colors[i % len(self.plot_initializer.colors)]
            
            # Plot reference solution if enabled for this run
            if data.reference and data.values_time_ref is not None:
                # Use initializer for reference time line
                self.reference_time_line = self.plot_initializer.create_reference_time_line(self.ax_time)
                # Set data immediately for static plotting - this line is NOT animated
                self.reference_time_line.set_data(data.values_time_ref, np.rad2deg(data.values_angle_ref))
                
            # Initialize line and marker for current simulation run
            line, marker = self.plot_initializer.create_time_plot_artists(self.ax_time, name, color)
            self.time_lines.append(line)
            self.time_markers.append(marker)
            
            # Update plot limits based on all data
            min_angle_deg = min(min_angle_deg, np.min(np.rad2deg(data.values_angle)))
            max_angle_deg = max(max_angle_deg, np.max(np.rad2deg(data.values_angle)))
            min_time = min(min_time, np.min(data.values_time))
            max_time = max(max_time, np.max(data.values_time))

        # Delegate axis setup to the initializer
        self.plot_initializer.setup_time_axis(self.ax_time, min_time, max_time, min_angle_deg, max_angle_deg)


        # --- Right: Pendulum animation ---
        # Delegate axis setup to the initializer
        self.plot_initializer.setup_pendulum_axis(self.ax_pend)
        
        # Initialize artists for EACH simulation run
        for i, (name, data) in enumerate(self.pendulum_data_runs.items()):
            color = self.plot_initializer.colors[i % len(self.plot_initializer.colors)]
            line, rect = self.plot_initializer.create_pendulum_artists(self.ax_pend, color)
            self.pendulum_lines.append(line)
            self.pendulum_rects.append(rect)
            
        # Initialize artists for the REFERENCE pendulum if it exists
        if self.reference_pendulum_data and self.reference_pendulum_data.values_angle_ref is not None:
            self.reference_pendulum_line, self.reference_pendulum_rect = \
                self.plot_initializer.create_reference_pendulum_artists(self.ax_pend)

        # Create pivot point (static)
        self.plot_initializer.create_pivot_point_artist(self.ax_pend)
        
        # Create time text artist
        self.text_pend = self.plot_initializer.create_time_text_artist(self.ax_pend)


    def _init_animation(self) -> Tuple:
        all_artists = []
        for line in self.time_lines:
            line.set_data([], [])
            all_artists.append(line)
        for marker in self.time_markers:
            marker.set_data([], [])
            all_artists.append(marker)
        for line in self.pendulum_lines:
            line.set_data([], [])
            all_artists.append(line)
        for rect in self.pendulum_rects:
            rect.set_transform(self.ax_pend.transData)
            all_artists.append(rect)
            
        # REMOVED: self.reference_time_line.set_data([], []) and appending it to all_artists
        # The reference_time_line is static and should not be cleared by _init_animation
        # or returned by _update_animation when blit=True.

        if self.reference_pendulum_line:
            self.reference_pendulum_line.set_data([], [])
            all_artists.append(self.reference_pendulum_line)
        if self.reference_pendulum_rect:
            self.reference_pendulum_rect.set_transform(self.ax_pend.transData)
            all_artists.append(self.reference_pendulum_rect)

        self.text_pend.set_text("")
        all_artists.append(self.text_pend)

        return tuple(all_artists)
        
    def _update_animation(self, frame: int) -> Tuple:
        all_artists = []

        # Update time text based on the first run's time
        if self.pendulum_data_runs:
            first_run_data = next(iter(self.pendulum_data_runs.values()))
            time_text_frame_idx = min(frame, len(first_run_data.values_time) - 1)
            self.text_pend.set_text(f"{first_run_data.values_time[time_text_frame_idx]:.2f} s")
            all_artists.append(self.text_pend)

        # Update each simulation run's pendulum animation
        for i, (name, data) in enumerate(self.pendulum_data_runs.items()):
            current_frame = min(frame, len(data.values_time) - 1)

            # Update time plot
            self.time_lines[i].set_data(data.values_time[:current_frame], np.rad2deg(data.values_angle[:current_frame]))
            all_artists.append(self.time_lines[i])
            
            # Update time marker
            self.time_markers[i].set_data([data.values_time[current_frame]], [np.rad2deg(data.values_angle[current_frame])])
            all_artists.append(self.time_markers[i])
            
            # Data for pendulum line and bob
            length_string = (self.plot_initializer.length_pend - self.plot_initializer.length_rect_long/2)
            x: float = self.plot_initializer.length_pend * np.sin(data.values_angle[current_frame])
            y: float = -self.plot_initializer.length_pend * np.cos(data.values_angle[current_frame])

            # Pendulum line
            x_line: float = x * length_string / self.plot_initializer.length_pend
            y_line: float = y * length_string / self.plot_initializer.length_pend
            self.pendulum_lines[i].set_data([0, x_line], [0, y_line])
            all_artists.append(self.pendulum_lines[i])
            
            # Rotate and move rectangle bob
            angle: float = np.arctan2(y, x)
            trans = Affine2D().rotate(angle).translate(x, y) + self.ax_pend.transData
            self.pendulum_rects[i].set_transform(trans)
            all_artists.append(self.pendulum_rects[i])
        
        # Update the reference pendulum animation (only the pendulum part)
        if self.reference_pendulum_data and self.reference_pendulum_data.values_angle_ref is not None:
            ref_data = self.reference_pendulum_data
            ref_current_frame = min(frame, len(ref_data.values_time_ref) - 1) 

            length_string_ref = (self.plot_initializer.length_pend - self.plot_initializer.length_rect_long/2)
            x_ref: float = self.plot_initializer.length_pend * np.sin(ref_data.values_angle_ref[ref_current_frame])
            y_ref: float = -self.plot_initializer.length_pend * np.cos(ref_data.values_angle_ref[ref_current_frame]) 

            x_line_ref: float = x_ref * length_string_ref / self.plot_initializer.length_pend
            y_line_ref: float = y_ref * length_string_ref / self.plot_initializer.length_pend
            if self.reference_pendulum_line:
                self.reference_pendulum_line.set_data([0, x_line_ref], [0, y_line_ref])
                all_artists.append(self.reference_pendulum_line)
            
            angle_ref: float = np.arctan2(y_ref, x_ref)
            trans_ref = Affine2D().rotate(angle_ref).translate(x_ref, y_ref) + self.ax_pend.transData
            if self.reference_pendulum_rect:
                self.reference_pendulum_rect.set_transform(trans_ref)
                all_artists.append(self.reference_pendulum_rect)
            
        return tuple(all_artists)

    # --- Public Methods ---
    def animate(
        self,
        repeat: bool = False
    ) -> FuncAnimation:
        
        self._create_animation_figure()
        sec_to_millisec: float = 1000

        max_frames = 0
        if self.pendulum_data_runs:
            max_frames = max(len(data.values_time) for data in self.pendulum_data_runs.values())
        if self.reference_pendulum_data and self.reference_pendulum_data.values_time_ref is not None:
            max_frames = max(max_frames, len(self.reference_pendulum_data.values_time_ref))

        self.ani = FuncAnimation(
            self.fig,
            self._update_animation,
            frames=max_frames,
            init_func=self._init_animation,
            blit=True,
            interval=self.animation_step_width * sec_to_millisec,
            repeat=repeat
        )
        plt.suptitle("Pendulum Animation with Time-Trace Indicator")
        plt.show()
        return self.ani

    def plot(self):
        """
        Visualizes the results of the pendulum simulation(s) in a static plot.
        """
        plt.figure(figsize=(10, 6))

        for i, (name, data) in enumerate(self.pendulum_data_runs.items()):
            color = self.plot_initializer.colors[i % len(self.plot_initializer.colors)]
            plt.plot(data.values_time, np.rad2deg(data.values_angle), color=color, label=f'Integrator ({name})', alpha=self.plot_initializer.alpha_value)
            
            if data.reference and data.values_time_ref is not None:
                plt.plot(data.values_time_ref, np.rad2deg(data.values_angle_ref), 
                         color=self.plot_initializer.reference_color, linestyle='--', label=f'Reference (RK45)', alpha=self.plot_initializer.alpha_value)
        
        plt.title("Pendulum simulation: Angle over time")
        plt.xlabel("Time/ s")
        plt.ylabel("Angle/ deg")
        plt.grid(True)
        plt.legend()
        plt.show()