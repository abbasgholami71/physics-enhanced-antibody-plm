import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Define the directories containing your simulation data
base_dir = Path("E:/ABBAS/physics-enhanced-antibody-plm")
sim_dirs = {
    "Original": base_dir / "results/MD/original",
    "Rank 1": base_dir / "results/MD/rank1",
    "Rank 2": base_dir / "results/MD/rank2",
    "Rank 3": base_dir / "results/MD/rank3"
}
output_dir = base_dir / "results/MD/comparison"

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

def parse_xvg(file_path):
    """Parse an XVG file and return time and RMSD data."""
    time = []
    rmsd = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comment and header lines
            if line.startswith('#') or line.startswith('@'):
                continue
                
            # Parse data lines
            values = line.split()
            if len(values) >= 2:
                time.append(float(values[0]))  # Time in ps
                rmsd.append(float(values[1]))  # RMSD in nm
    
    return np.array(time), np.array(rmsd)

def parse_interface_residues_xvg(file_path):
    """Parse interface residues XVG file and return residue numbers and probabilities."""
    residue_numbers = []
    probabilities = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comment and header lines
            if line.startswith('#') or line.startswith('@'):
                continue
                
            # Parse data lines
            values = line.split()
            if len(values) >= 2:
                residue_numbers.append(int(float(values[0])))  # Residue number
                probabilities.append(float(values[1]))  # Probability
    
    return np.array(residue_numbers), np.array(probabilities)

def parse_energy_xvg(file_path):
    """Parse energy XVG file and return time and energy data."""
    time = []
    energy = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comment and header lines
            if line.startswith('#') or line.startswith('@'):
                continue
                
            # Parse data lines
            values = line.split()
            if len(values) >= 2:
                time.append(float(values[0]))  # Time in ps
                energy.append(float(values[1]))  # Energy
    
    return np.array(time), np.array(energy)

def parse_hbonds_xvg(file_path):
    """Parse hydrogen bonds XVG file and return time and number of hydrogen bonds."""
    time = []
    hbonds = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comment and header lines
            if line.startswith('#') or line.startswith('@'):
                continue
                
            # Parse data lines
            values = line.split()
            if len(values) >= 2:
                time.append(float(values[0]))  # Time in ps
                hbonds.append(float(values[1]))  # Number of hydrogen bonds
    
    return np.array(time), np.array(hbonds)

# Colors for different simulations
colors = ["blue", "green", "red", "purple"]

# PART 1: Plot RMSD data for each simulation separately
print("Generating RMSD plots...")

for i, (sim_name, sim_dir) in enumerate(sim_dirs.items()):
    rmsd_file = sim_dir / "rmsd.xvg"
    
    if rmsd_file.exists():
        time, rmsd = parse_xvg(rmsd_file)
        
        # Create a new figure for each simulation
        plt.figure(figsize=(10, 6))
        
        # Plot the data
        plt.plot(time, rmsd, 
                 color=colors[i], 
                 linewidth=2)
        
        # Add plot details
        plt.title(f"RMSD for {sim_name} CR3022+RBD Simulation", fontsize=14)
        plt.xlabel("Time (ps)", fontsize=12)
        plt.ylabel("RMSD (nm)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Make the plot look nicer
        plt.tight_layout()
        
        # Save the figure as PNG only
        output_file = output_dir / f"rmsd_{sim_name.lower().replace(' ', '_')}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"RMSD plot for {sim_name} saved to {output_file}")
        
        # Close the figure to free memory
        plt.close()
    else:
        print(f"Warning: RMSD file not found for {sim_name} at {rmsd_file}")

# Also create a combined plot with all simulations for comparison
plt.figure(figsize=(12, 8))

for i, (sim_name, sim_dir) in enumerate(sim_dirs.items()):
    rmsd_file = sim_dir / "rmsd.xvg"
    
    if rmsd_file.exists():
        time, rmsd = parse_xvg(rmsd_file)
        plt.plot(time, rmsd, 
                 label=sim_name, 
                 color=colors[i],
                 linewidth=2)

# Add plot details for combined plot
plt.title("RMSD Comparison of All CR3022+RBD Simulations", fontsize=14)
plt.xlabel("Time (ps)", fontsize=12)
plt.ylabel("RMSD (nm)", fontsize=12)
plt.legend(loc="best", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)

# Make the plot look nicer
plt.tight_layout()

# Save the combined figure as PNG only
combined_output_file = output_dir / "rmsd_all_simulations.png"
plt.savefig(combined_output_file, dpi=300, bbox_inches='tight')
print(f"Combined RMSD plot saved to {combined_output_file}")

# Close the figure
plt.close()

print("All RMSD plots generated successfully.")

# PART 2: Compare interface residues
print("\nComparing interface residues...")

# Define chains to analyze
chains = ["HC", "LC"]  # Heavy Chain and Light Chain

for chain in chains:
    # Create a figure for this chain's interface residues
    plt.figure(figsize=(14, 8))
    
    # Store max probability for y-axis scaling
    max_prob = 0
    
    # Process each simulation
    for i, (sim_name, sim_dir) in enumerate(sim_dirs.items()):
        interface_file = sim_dir / f"interface_residues_{chain}.xvg"
        
        if interface_file.exists():
            residue_numbers, probabilities = parse_interface_residues_xvg(interface_file)
            
            # Update max probability if needed
            current_max = np.max(probabilities) if len(probabilities) > 0 else 0
            max_prob = max(max_prob, current_max)
            
            # Create bar plot for each simulation with slight offset for better visibility
            offset = i * 0.2 - 0.3  # Creates offsets like -0.3, -0.1, 0.1, 0.3
            width = 0.18  # Width of bars
            
            plt.bar(residue_numbers + offset, probabilities, 
                   width=width, 
                   color=colors[i], 
                   alpha=0.7, 
                   label=sim_name)
            
            print(f"Processed {chain} interface residues for {sim_name}")
        else:
            print(f"Warning: Interface residue file not found for {sim_name} at {interface_file}")
    
    # Add plot details
    plt.title(f"{chain} Interface Residues Probability", fontsize=14)
    plt.xlabel("Residue Number", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    plt.legend(loc="best", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Set y-axis limit with some padding
    plt.ylim(0, max_prob * 1.1)
    
    # If there are many residues, improve x-axis readability
    if len(plt.gca().get_xticks()) > 15:
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save the figure
    output_file = output_dir / f"interface_residues_{chain}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"{chain} interface residues plot saved to {output_file}")
    
    plt.close()

# Also create a heatmap version for easier visualization
print("\nGenerating interface residues heatmaps...")

for chain in chains:
    # Collect data for all simulations
    all_data = {}
    all_residues = set()
    
    for sim_name, sim_dir in sim_dirs.items():
        interface_file = sim_dir / f"interface_residues_{chain}.xvg"
        
        if interface_file.exists():
            residue_numbers, probabilities = parse_interface_residues_xvg(interface_file)
            all_data[sim_name] = {res: prob for res, prob in zip(residue_numbers, probabilities)}
            all_residues.update(residue_numbers)
    
    if not all_data:
        print(f"No interface residue data found for {chain}")
        continue
    
    # Convert to a dataframe for heatmap plotting
    all_residues = sorted(list(all_residues))
    data_for_heatmap = np.zeros((len(sim_dirs), len(all_residues)))
    
    for i, sim_name in enumerate(sim_dirs.keys()):
        if sim_name in all_data:
            for j, res in enumerate(all_residues):
                data_for_heatmap[i, j] = all_data[sim_name].get(res, 0)
    
    # Create heatmap
    plt.figure(figsize=(16, 6))
    plt.imshow(data_for_heatmap, aspect='auto', cmap='viridis')
    
    # Add colorbar and labels
    cbar = plt.colorbar(label='Probability')
    plt.title(f"{chain} Interface Residues Probability Heatmap", fontsize=14)
    plt.xlabel("Residue Number", fontsize=12)
    plt.ylabel("Simulation", fontsize=12)
    
    # Set x-ticks to residue numbers (with reduced frequency if too many)
    step = max(1, len(all_residues) // 20)  # Show at most ~20 ticks
    plt.xticks(range(0, len(all_residues), step), [all_residues[i] for i in range(0, len(all_residues), step)])
    
    # Set y-ticks to simulation names
    plt.yticks(range(len(sim_dirs)), list(sim_dirs.keys()))
    
    plt.tight_layout()
    
    # Save the heatmap
    output_file = output_dir / f"interface_residues_{chain}_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"{chain} interface residues heatmap saved to {output_file}")
    
    plt.close()

# PART 3: Plot interaction energies
print("\nPlotting interaction energies...")

# Create a figure for the interaction energies
plt.figure(figsize=(12, 8))

for i, (sim_name, sim_dir) in enumerate(sim_dirs.items()):
    energy_file = sim_dir / "interaction_energy.xvg"
    
    if energy_file.exists():
        time, energy = parse_energy_xvg(energy_file)
        
        # Convert to positive energy values if needed
        # Assuming interaction energies are negative by convention
        positive_energy = np.abs(energy)
        
        plt.plot(time, positive_energy, 
                label=sim_name, 
                color=colors[i],
                linewidth=2)
        
        print(f"Processed interaction energy for {sim_name}")
    else:
        print(f"Warning: Interaction energy file not found for {sim_name} at {energy_file}")

# Add plot details
plt.title("Antibody-Antigen Interaction Energy Comparison", fontsize=14)
plt.xlabel("Time (ps)", fontsize=12)
plt.ylabel("Interaction Energy (absolute value, kJ/mol)", fontsize=12)
plt.legend(loc="best", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()

# Save the figure
energy_output_file = output_dir / "interaction_energy_comparison.png"
plt.savefig(energy_output_file, dpi=300, bbox_inches='tight')
print(f"Interaction energy comparison plot saved to {energy_output_file}")

plt.close()

# PART 4: Plot number of hydrogen bonds
print("\nPlotting hydrogen bond numbers...")

# Create a figure for the hydrogen bonds
plt.figure(figsize=(12, 8))

for i, (sim_name, sim_dir) in enumerate(sim_dirs.items()):
    hbonds_file = sim_dir / "hbnum_RBD_Antibody.xvg"
    
    if hbonds_file.exists():
        time, hbonds = parse_hbonds_xvg(hbonds_file)
        
        plt.plot(time, hbonds, 
                label=sim_name, 
                color=colors[i],
                linewidth=2)
        
        print(f"Processed hydrogen bonds for {sim_name}")
    else:
        print(f"Warning: Hydrogen bonds file not found for {sim_name} at {hbonds_file}")

# Add plot details
plt.title("Number of Hydrogen Bonds between Antibody and RBD", fontsize=14)
plt.xlabel("Time (ps)", fontsize=12)
plt.ylabel("Number of Hydrogen Bonds", fontsize=12)
plt.legend(loc="best", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)

# Add a horizontal line at y=0 for clarity
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

plt.tight_layout()

# Save the figure
hbonds_output_file = output_dir / "hydrogen_bonds_comparison.png"
plt.savefig(hbonds_output_file, dpi=300, bbox_inches='tight')
print(f"Hydrogen bonds comparison plot saved to {hbonds_output_file}")

plt.close()

print("\nAll plots generated successfully.")