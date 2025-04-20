import numpy as np
import pandas as pd
from data_generation.driving_agent import Agent, Road
import random
from tqdm import tqdm

def calculate_max_lateral_force(mass_slugs):
    # Calculate g-force based on mass using linear interpolation
    # For 95 slugs: 0.9g, for 625 slugs: 0.5g
    g_force = 0.9 - ((mass_slugs - 95) / (625 - 95)) * (0.9 - 0.5)
    
    # Calculate force in pounds-force (lbf)
    # 1 slug * 1 ft/s² = 1 lbf
    # For g-force, we use 32.174 ft/s² (standard gravity)
    force_lbf = mass_slugs * g_force * 32.174
    
    return force_lbf

def run_simulation(sim_id, dt=0.1):
    # Randomly sample parameters based on comments in data_prep.py
    # Mass: 95-625 slugs
    mass = random.uniform(95, 625)
    
    # Calculate max lateral force based on mass
    max_lateral_force = calculate_max_lateral_force(mass)
    
    # Road width: 12 feet (standard lane width)
    road_width = 12
    
    # Speed limit: 45 mph (typical urban speed limit)
    speed_limit = random.uniform(5, 14) * 5
    
    # Impairment level: random between 0 and 1
    random_val = random.uniform(0, 1.5)
    impairment_level = random_val / 3 if random_val < 0.75 else random_val - 0.5
    
    # Create road and agent
    road = Road(width=road_width, speed_limit=speed_limit)
    agent = Agent(road, impairment_level=impairment_level, mass=mass, max_lateral_force=max_lateral_force)
    
    # Run simulation and get trajectory
    trajectory = agent.run(dt)
    
    # Convert to DataFrame
    data = []
    time = 0
    for x, y in trajectory:
        data.append({
            'id': sim_id,
            'x': x,
            'y': y,
            'time_from_start': time,
            'impaired': 'yes' if impairment_level > 0.25 else 'no'
        })
        time += dt
    
    return pd.DataFrame(data)

def main():
    # Run 1000 simulations
    all_data = []
    for sim_id in tqdm(range(1, 10001), desc="Running simulations"):
        df = run_simulation(sim_id)
        all_data.append(df)
    
    # Combine all data
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Save to CSV
    final_df.to_csv('simulation_data.csv', index=False)
    print(f"Data saved to simulation_data.csv with {len(final_df)} rows")

if __name__ == "__main__":
    main() 