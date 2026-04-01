import wntr
import numpy as np

wn = wntr.network.WaterNetworkModel('base.inp')
sim = wntr.sim.EpanetSimulator(wn)
results = sim.run_sim()

# Restrict to junctions only (excludes reservoir and tank nodes)
junction_pressures = results.node['pressure'][wn.junction_name_list]

# Find the timestep where the average junction pressure is lowest (peak demand)
peak_time = junction_pressures.mean(axis=1).idxmin()

# Critical point: junction with lowest pressure at peak demand time
critical_point = junction_pressures.loc[peak_time].idxmin()

print(f"Critical Point: Node {critical_point}, Pressure: {junction_pressures.loc[peak_time, critical_point]:.2f} m")

wn = wntr.network.WaterNetworkModel('base.inp')
sim = wntr.sim.EpanetSimulator(wn)
results = sim.run_sim()

# Restrict to junctions only (excludes reservoir)
junction_pressures = results.node['pressure'][wn.junction_name_list]

# Calculate the average pressure across all junctions at each timestep
avg_pressure = junction_pressures.mean(axis=1)

# For each junction, compute the mean absolute deviation from the zone average over all timesteps
deviations = (junction_pressures.subtract(avg_pressure, axis=0)).abs().mean()

# The average zonal point is the junction with the smallest deviation
avg_zonal_point = deviations.idxmin()

print(f"Average Zonal Point: Node {avg_zonal_point}, Mean Deviation: {deviations[avg_zonal_point]:.4f} m")