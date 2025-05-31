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

matplotlib.use('TkAgg')


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

    text_pend: matplotlib.text.Text # Assuming one global time text
    ani: Optional[FuncAnimation]

    # --- Data and styling ---
    # This will now store a dictionary of PendulumData objects, keyed by their name
    pendulum_data_runs: Dict[str, PendulumData] 
    
    # Constants for pendulum size
    length_pend: float = 1.0
    length_rect_long = 0.3
    length_rect_short = 0.2
    
    # Color palette for different simulation runs
    # Using distinct colors for clarity
    colors: List[np.ndarray] = [
        np.array([47/255, 82/255, 143/255]),  # A shade of blue
        np.array([200/255, 50/255, 50/255]), # A shade of red
        np.array([50/255, 150/255, 50/255]), # A shade of green
        np.array([255/255, 165/255, 0/255]) # Orange
    ]
    reference_color: str = 'gray'

    # --- Constructor ---
    def __init__(self, simulation_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
                 reference: bool = False, omega_0_ref: float = np.sqrt(9.81),
                 D_ref: float = 0.05) -> None:
        """
        Initializes the pendulum visualization by first processing the data for multiple runs.

        Args:
            simulation_results (Dict[str, Tuple[np.ndarray, np.ndarray]]): 
                A dictionary where keys are names (e.g., 'Euler', 'RK2') and values are 
                tuples (time_values_array, state_values_array) for each simulation run.
            reference (bool): Whether to compute and show a single reference solution 
                              (applied to the first simulation run in the dictionary).
            omega_0_ref (float): Natural frequency for the reference solution.
            D_ref (float): Damping coefficient for the reference solution.
        """
        self.pendulum_data_runs = {}
        # Get the first key to decide which run gets the reference (if 'reference' is True)
        # This assumes the order of dictionary items is consistent (Python 3.7+ guarantee)
        run_names = list(simulation_results.keys())
        first_run_name = run_names[0] if run_names else None

        for name, (t_vals, u_vals) in simulation_results.items():
            # Only apply 'reference' flag to the first simulation, or adjust logic if needed
            run_reference = reference if name == first_run_name else False
            
            self.pendulum_data_runs[name] = PendulumData(
                values_time=t_vals,
                values_state=u_vals,
                reference=run_reference,
                omega_0_ref=omega_0_ref,
                D_ref=D_ref
            )
        
        # Determine the animation interval based on the smallest step_width among runs
        # or just use the first run's step_width for simplicity.
        self.animation_step_width = min(
            data.step_width for data in self.pendulum_data_runs.values()
        ) if self.pendulum_data_runs else 0.01 # Fallback if no runs

    # --- Animation Setup ---
    def _create_animation_figure(self) -> None:
        self.fig, (self.ax_time, self.ax_pend) = plt.subplots(1, 2, figsize=(12, 6))

        # Initialize lists for matplotlib artists
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
            color = self.colors[i % len(self.colors)] # Cycle through colors
            
            # Plot reference solution if enabled for this run
            if data.reference:
                self.ax_time.plot(data.values_time_ref, np.rad2deg(data.values_angle_ref), 
                                  color=self.reference_color, linestyle='--', label=f'Reference ({name})')

            # Initialize line and marker for current simulation run
            line, = self.ax_time.plot([], [], lw=1, color=color, label=f'Integrator ({name})')
            marker, = self.ax_time.plot(data.values_time[0], np.rad2deg(data.values_angle[0]), 'o', ms=8, color=color)
            
            self.time_lines.append(line)
            self.time_markers.append(marker)
            
            # Update plot limits based on all data
            min_angle_deg = min(min_angle_deg, np.min(np.rad2deg(data.values_angle)))
            max_angle_deg = max(max_angle_deg, np.max(np.rad2deg(data.values_angle)))
            min_time = min(min_time, np.min(data.values_time))
            max_time = max(max_time, np.max(data.values_time))


        self.ax_time.set_title("Winkel über Zeit")
        self.ax_time.set_xlabel("Zeit / s")
        self.ax_time.set_ylabel("Winkel / grad")
        self.ax_time.set_xlim(min_time * 0.8, max_time * 1.05)
        self.ax_time.set_ylim(min_angle_deg * 1.2, max_angle_deg * 1.2)
        self.ax_time.grid(True)
        self.ax_time.legend()

        # --- Right: Pendulum animation ---
        self.ax_pend.set_xlim(-self.length_pend * 1.2, self.length_pend * 1.2)
        self.ax_pend.set_ylim(-self.length_pend * 1.2, self.length_pend * 1.2)
        self.ax_pend.set_aspect('equal')
        self.ax_pend.set_xlabel('')
        self.ax_pend.set_ylabel('')
        self.ax_pend.grid(True, which="both")

        self.ax_pend.tick_params(
            axis="both",
            which="both",
            length=0,
            labelbottom=False,
            labelleft=False
        )

        for i, (name, data) in enumerate(self.pendulum_data_runs.items()):
            color = self.colors[i % len(self.colors)]
            line, = self.ax_pend.plot([], [], lw=2, color=color, linestyle='-') # Use color for pend line
            self.pendulum_lines.append(line)

            # Initialize rectangle bob
            rect = Rectangle(
                (-self.length_rect_long/2, -self.length_rect_short/2),
                self.length_rect_long,
                self.length_rect_short,
                facecolor=color, # Use color for bob
                edgecolor='black',
                linewidth=2,
                transform=self.ax_pend.transData
            )
            self.ax_pend.add_patch(rect)
            self.pendulum_rects.append(rect)
            
        self.ax_pend.scatter(0, 0, s=200, c='black', marker='o', zorder=10) # Fixed pivot point
        
        # Time text (can be global or per pendulum) - keeping it global for simplicity
        self.text_pend = self.ax_pend.text(
            0.5, 0.95,
            "",
            transform=self.ax_pend.transAxes,
            ha="center",
            va="top",
            fontsize=12
        )
        self.ax_pend.set_title("Pendel-Animation")


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
            rect.set_transform(self.ax_pend.transData) # Reset transform
            all_artists.append(rect)
            
        self.text_pend.set_text("")
        all_artists.append(self.text_pend) # Add text to artists to be returned

        return tuple(all_artists)
        
    def _update_animation(self, frame: int) -> Tuple:
        all_artists = []

        if self.pendulum_data_runs:
            first_run_data = next(iter(self.pendulum_data_runs.values()))
            # Apply clamping to the frame index used for the time text
            time_text_frame_idx = min(frame, len(first_run_data.values_time) - 1)
            self.text_pend.set_text(f"{first_run_data.values_time[time_text_frame_idx]:.2f} s")
            all_artists.append(self.text_pend)

        for i, (name, data) in enumerate(self.pendulum_data_runs.items()):
            # Ensure frame is within bounds for this specific run's data
            current_frame = min(frame, len(data.values_time) - 1)

            # Update time plot
            self.time_lines[i].set_data(data.values_time[:current_frame], np.rad2deg(data.values_angle[:current_frame]))
            all_artists.append(self.time_lines[i])
            
            # Update time marker
            self.time_markers[i].set_data([data.values_time[current_frame]], [np.rad2deg(data.values_angle[current_frame])])
            all_artists.append(self.time_markers[i])
            
            # Data for pendulum line and bob
            length_string = (self.length_pend - self.length_rect_long/2)
            x: float = self.length_pend * np.sin(data.values_angle[current_frame])
            y: float = -self.length_pend * np.cos(data.values_angle[current_frame])

            # Pendulum line
            x_line: float = x * length_string / self.length_pend
            y_line: float = y * length_string / self.length_pend
            self.pendulum_lines[i].set_data([0, x_line], [0, y_line])
            all_artists.append(self.pendulum_lines[i])
            
            # Rotate and move rectangle bob
            angle: float = np.arctan2(y, x)
            trans = Affine2D().rotate(angle).translate(x, y) + self.ax_pend.transData
            self.pendulum_rects[i].set_transform(trans)
            all_artists.append(self.pendulum_rects[i])
            
        return tuple(all_artists)

    # --- Public Methods ---
    def animate(
        self,
        repeat: bool = False
    ) -> FuncAnimation:
        
        self._create_animation_figure()
        convert_sec_to_millisec: float = 1000

        # Animation frames will be based on the longest simulation run,
        # or the one with the highest number of data points.
        max_frames = max(len(data.values_time) for data in self.pendulum_data_runs.values()) if self.pendulum_data_runs else 0

        self.ani = FuncAnimation(
            self.fig,
            self._update_animation,
            frames=max_frames, # Ensure animation runs for the full duration of longest run
            init_func=self._init_animation,
            blit=True,
            interval=self.animation_step_width * convert_sec_to_millisec,
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
            color = self.colors[i % len(self.colors)]
            plt.plot(data.values_time, np.rad2deg(data.values_angle), color=color, label=f'Integrator ({name})')
            
            if data.reference:
                plt.plot(data.values_time_ref, np.rad2deg(data.values_angle_ref), 
                         color=self.reference_color, linestyle='--', label=f'Reference ({name})')
        
        plt.title("Pendelsimulation: Winkel über Zeit")
        plt.xlabel("Zeit/ s")
        plt.ylabel("Winkel/ deg")
        plt.grid(True)
        plt.legend()
        plt.show()