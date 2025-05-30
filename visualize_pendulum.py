# plot_funktionen.py
import numpy as np # Falls für Typ-Annotationen oder Beispiele benötigt
import matplotlib
matplotlib.use('TkAgg') # Oder 'Qt5Agg', 'QtAgg'. Probiere ggf. andere aus.
import matplotlib.pyplot as plt


def plot_simulation_data(time_values, angle_values_deg, integrator_name="Euler explizit"):
    """
    Visualisiert die Ergebnisse der Pendelsimulation.
    (Deine bekannte Plot-Funktion)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(time_values, angle_values_deg, label=f"{integrator_name}")

    plt.title("Pendelsimulation: Winkel über Zeit")

    plt.xlabel("Zeit/ s")
    plt.ylabel("Winkel/ deg")
    plt.grid(True)
    plt.legend()
    plt.show()






