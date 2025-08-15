# scripts/colabfold_interface_analysis.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import subprocess
import requests
import time
import io
import zipfile
from Bio.PDB import PDBParser, NeighborSearch

# Define directories
RESULTS_DIR = 'results/structure_prediction'
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
DATA_DIR = os.path.join(RESULTS_DIR, 'data')
STRUCTURE_DIR = os.path.join(RESULTS_DIR, 'structures')

# Create necessary directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STRUCTURE_DIR, exist_ok=True)

print("Starting structure-based interface analysis...")

# Load sequences
print("Loading sequences...")
with open('data/heavy_chain.fasta', 'r') as f:
    lines = f.readlines()
    heavy_seq = lines[1].strip()

with open('data/light_chain.fasta', 'r') as f:
    lines = f.readlines()
    light_seq = lines[1].strip()
    
with open('data/rbd.fasta', 'r') as f:
    lines = f.readlines()
    rbd_seq = lines[1].strip()

print(f"Heavy chain sequence length: {len(heavy_seq)}")
print(f"Light chain sequence length: {len(light_seq)}")
print(f"RBD sequence length: {len(rbd_seq)}")

# Create FASTA files for ColabFold input
def create_fasta(filename, sequences):
    """Create a FASTA file with multiple sequences"""
    with open(filename, 'w') as f:
        for name, seq in sequences.items():
            f.write(f">{name}\n{seq}\n")
    return filename

# Create FASTA for antibody-antigen complex
complex_fasta = os.path.join(DATA_DIR, 'complex.fasta')
create_fasta(complex_fasta, {
    'H': heavy_seq,  # Heavy chain
    'L': light_seq,  # Light chain
    'R': rbd_seq     # RBD
})

# Instructions for ColabFold
print("""
=================================================================
COLABFOLD STRUCTURE PREDICTION INSTRUCTIONS

Since AlphaFold2 is complex to install locally, you can use ColabFold instead:

1. Go to https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb

2. Upload the FASTA file from:
   {fasta_path}

3. Set these parameters:
   - Use MSA: True
   - Use templates: False
   - Number of models: 5
   - Model type: AlphaFold2-multimer
   - Pair mode: unpaired+paired

4. Run the notebook to predict the structure

5. Download the resulting structures and place the highest-ranked PDB file at:
   {output_path}

6. Then rerun this script to continue the analysis
=================================================================
""".format(
    fasta_path=complex_fasta,
    output_path=os.path.join(STRUCTURE_DIR, 'predicted_complex.pdb')
))

# Check if structure file already exists
structure_file = os.path.join(STRUCTURE_DIR, 'predicted_complex.pdb')
if not os.path.exists(structure_file):
    user_input = input("Press Enter when you've placed the predicted structure file, or type 'skip' to use a placeholder structure: ")
    if user_input.lower() == 'skip':
        # Create a placeholder structure file for testing
        print("Using placeholder structure file")
        with open(structure_file, 'w') as f:
            f.write("PLACEHOLDER STRUCTURE FILE\n")
            f.write("Please replace with actual predicted structure from ColabFold\n")
    else:
        if not os.path.exists(structure_file):
            print(f"Error: Structure file not found at {structure_file}")
            print("Please make sure the file exists and rerun the script")
            exit(1)

# Function to analyze the interface from a structure file
def analyze_interface(structure_path):
    """
    Analyze the interface between antibody and antigen in the predicted structure
    """
    print(f"Analyzing interface from structure: {structure_path}")
    
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("complex", structure_path)
        model = structure[0]  # First model
        
        # Extract chains
        available_chains = [chain.id for chain in model.get_chains()]
        print(f"Available chains in structure: {available_chains}")
        
        # Check the number of chains
        if len(available_chains) < 3:
            print(f"Warning: Expected 3 chains (Heavy, Light, RBD) but found {len(available_chains)}")
            
            if len(available_chains) == 1:
                print("Single chain found. Attempting to separate chains based on residue numbering...")
                
                # Get the only chain
                chain = list(model.get_chains())[0]
                
                # Try to separate by residue gaps or by known lengths
                residues = list(chain.get_residues())
                
                # Get all residue IDs
                res_ids = [res.id[1] for res in residues]
                
                # Look for large gaps in residue numbering
                gaps = []
                for i in range(1, len(res_ids)):
                    if res_ids[i] - res_ids[i-1] > 50:  # Large gap threshold
                        gaps.append(i)
                
                if len(gaps) >= 2:
                    print(f"Found potential chain boundaries at residues: {[res_ids[i] for i in gaps]}")
                    
                    # Split into 3 chains using the gaps
                    heavy_residues = residues[0:gaps[0]]
                    light_residues = residues[gaps[0]:gaps[1]]
                    rbd_residues = residues[gaps[1]:]
                    
                    print(f"Split into Heavy: {len(heavy_residues)} residues, "
                          f"Light: {len(light_residues)} residues, "
                          f"RBD: {len(rbd_residues)} residues")
                else:
                    # Use predefined lengths from the sequences
                    print("Using predefined sequence lengths to separate chains")
                    
                    total_len = len(residues)
                    h_len = len(heavy_seq)
                    l_len = len(light_seq)
                    r_len = len(rbd_seq)
                    
                    if total_len >= h_len + l_len + r_len:
                        heavy_residues = residues[0:h_len]
                        light_residues = residues[h_len:h_len+l_len]
                        rbd_residues = residues[h_len+l_len:h_len+l_len+r_len]
                        
                        print(f"Split into Heavy: {len(heavy_residues)} residues, "
                              f"Light: {len(light_residues)} residues, "
                              f"RBD: {len(rbd_residues)} residues")
                    else:
                        raise ValueError("Cannot split single chain: not enough residues")
            else:
                # Try to identify chains by length
                chains = list(model.get_chains())
                chain_lengths = [len(list(chain.get_residues())) for chain in chains]
                
                # Use sequence lengths to find the best match for each chain
                h_len = len(heavy_seq)
                l_len = len(light_seq)
                r_len = len(rbd_seq)
                
                # Find the best match for each chain
                length_diffs = []
                for i, chain_len in enumerate(chain_lengths):
                    h_diff = abs(chain_len - h_len)
                    l_diff = abs(chain_len - l_len)
                    r_diff = abs(chain_len - r_len)
                    length_diffs.append((i, h_diff, l_diff, r_diff))
                
                # Sort by the smallest difference
                length_diffs.sort(key=lambda x: min(x[1], x[2], x[3]))
                
                # Assign chains
                chain_assignment = {}
                for i, h_diff, l_diff, r_diff in length_diffs:
                    min_diff = min(h_diff, l_diff, r_diff)
                    if min_diff == h_diff and 'heavy' not in chain_assignment:
                        chain_assignment['heavy'] = i
                    elif min_diff == l_diff and 'light' not in chain_assignment:
                        chain_assignment['light'] = i
                    elif min_diff == r_diff and 'rbd' not in chain_assignment:
                        chain_assignment['rbd'] = i
                
                if len(chain_assignment) == 3:
                    heavy_residues = list(chains[chain_assignment['heavy']].get_residues())
                    light_residues = list(chains[chain_assignment['light']].get_residues())
                    rbd_residues = list(chains[chain_assignment['rbd']].get_residues())
                    
                    print(f"Assigned chains by length: "
                          f"Heavy={available_chains[chain_assignment['heavy']]} ({len(heavy_residues)} residues), "
                          f"Light={available_chains[chain_assignment['light']]} ({len(light_residues)} residues), "
                          f"RBD={available_chains[chain_assignment['rbd']]} ({len(rbd_residues)} residues)")
                else:
                    raise ValueError("Cannot identify chains by length")
        else:
            # We have at least 3 chains
            chains = list(model.get_chains())
            heavy_chain = chains[0]
            light_chain = chains[1]
            rbd_chain = chains[2]
            
            print(f"Using first three chains: {chains[0].id}, {chains[1].id}, {chains[2].id}")
            
            heavy_residues = list(heavy_chain.get_residues())
            light_residues = list(light_chain.get_residues())
            rbd_residues = list(rbd_chain.get_residues())
        
        # Define distance threshold for interface
        INTERFACE_CUTOFF = 4.5  # Angstroms
        
        # Get all atoms in each chain
        heavy_atoms = [atom for res in heavy_residues for atom in res.get_atoms()]
        light_atoms = [atom for res in light_residues for atom in res.get_atoms()]
        rbd_atoms = [atom for res in rbd_residues for atom in res.get_atoms()]
        
        print(f"Found {len(heavy_atoms)} atoms in heavy chain")
        print(f"Found {len(light_atoms)} atoms in light chain")
        print(f"Found {len(rbd_atoms)} atoms in RBD chain")
        
        # Create NeighborSearch object for RBD
        ns_rbd = NeighborSearch(rbd_atoms)
        
        # Find interface residues
        heavy_interface = set()
        light_interface = set()
        rbd_interface = set()
        
        # Find heavy chain interface residues
        for atom in heavy_atoms:
            close_atoms = ns_rbd.search(atom.coord, INTERFACE_CUTOFF, 'A')
            if close_atoms:
                heavy_interface.add(atom.get_parent().id[1])  # Add residue number
                for close_atom in close_atoms:
                    rbd_interface.add(close_atom.get_parent().id[1])
        
        # Find light chain interface residues
        for atom in light_atoms:
            close_atoms = ns_rbd.search(atom.coord, INTERFACE_CUTOFF, 'A')
            if close_atoms:
                light_interface.add(atom.get_parent().id[1])  # Add residue number
                for close_atom in close_atoms:
                    rbd_interface.add(close_atom.get_parent().id[1])
        
        # Convert to sorted lists
        heavy_interface = sorted(list(heavy_interface))
        light_interface = sorted(list(light_interface))
        rbd_interface = sorted(list(rbd_interface))
        
        print(f"Found {len(heavy_interface)} interface residues in heavy chain")
        print(f"Found {len(light_interface)} interface residues in light chain")
        print(f"Found {len(rbd_interface)} interface residues in RBD")
        
        # Save interface residues
        with open(os.path.join(DATA_DIR, 'interface_residues.txt'), 'w') as f:
            f.write(f"Heavy chain interface residues: {heavy_interface}\n")
            f.write(f"Light chain interface residues: {light_interface}\n")
            f.write(f"RBD interface residues: {rbd_interface}\n")
        
        # Calculate detailed contact map
        print("Calculating detailed contact map...")
        
        # Initialize contact maps
        heavy_rbd_contacts = np.zeros((len(heavy_seq), len(rbd_seq)))
        light_rbd_contacts = np.zeros((len(light_seq), len(rbd_seq)))
        
        # Find minimum distance between residues
        def min_distance(res1_atoms, res2_atoms):
            min_dist = float('inf')
            for atom1 in res1_atoms:
                for atom2 in res2_atoms:
                    dist = np.linalg.norm(atom1.coord - atom2.coord)
                    min_dist = min(min_dist, dist)
            return min_dist
        
        # Collect residues by residue number
        heavy_residues = {}
        for res in heavy_chain.get_residues():
            res_id = res.id[1]
            if 1 <= res_id <= len(heavy_seq):  # Ensure valid index
                heavy_residues[res_id] = [atom for atom in res.get_atoms()]
        
        light_residues = {}
        for res in light_chain.get_residues():
            res_id = res.id[1]
            if 1 <= res_id <= len(light_seq):  # Ensure valid index
                light_residues[res_id] = [atom for atom in res.get_atoms()]
        
        rbd_residues = {}
        for res in rbd_chain.get_residues():
            res_id = res.id[1]
            if 1 <= res_id <= len(rbd_seq):  # Ensure valid index
                rbd_residues[res_id] = [atom for atom in res.get_atoms()]
        
        # Calculate contact maps
        for h_idx in range(len(heavy_seq)):
            h_res_id = h_idx + 1  # 1-indexed
            if h_res_id in heavy_residues:
                for r_idx in range(len(rbd_seq)):
                    r_res_id = r_idx + 1  # 1-indexed
                    if r_res_id in rbd_residues:
                        dist = min_distance(heavy_residues[h_res_id], rbd_residues[r_res_id])
                        heavy_rbd_contacts[h_idx, r_idx] = dist
        
        for l_idx in range(len(light_seq)):
            l_res_id = l_idx + 1  # 1-indexed
            if l_res_id in light_residues:
                for r_idx in range(len(rbd_seq)):
                    r_res_id = r_idx + 1  # 1-indexed
                    if r_res_id in rbd_residues:
                        dist = min_distance(light_residues[l_res_id], rbd_residues[r_res_id])
                        light_rbd_contacts[l_idx, r_idx] = dist
        
        # Convert distances to contact maps (1 for contact, 0 for no contact)
        CONTACT_CUTOFF = 8.0  # Wider cutoff for general contacts
        heavy_rbd_binary = np.zeros_like(heavy_rbd_contacts)
        light_rbd_binary = np.zeros_like(light_rbd_contacts)
        
        # Where distance is not 0 and less than cutoff, set to 1
        mask_h = (heavy_rbd_contacts > 0) & (heavy_rbd_contacts <= CONTACT_CUTOFF)
        mask_l = (light_rbd_contacts > 0) & (light_rbd_contacts <= CONTACT_CUTOFF)
        
        heavy_rbd_binary[mask_h] = 1
        light_rbd_binary[mask_l] = 1
        
        # Save contact maps
        np.save(os.path.join(DATA_DIR, 'heavy_rbd_contacts.npy'), heavy_rbd_contacts)
        np.save(os.path.join(DATA_DIR, 'light_rbd_contacts.npy'), light_rbd_contacts)
        np.save(os.path.join(DATA_DIR, 'heavy_rbd_binary.npy'), heavy_rbd_binary)
        np.save(os.path.join(DATA_DIR, 'light_rbd_binary.npy'), light_rbd_binary)
        
        # Calculate binding scores
        # Using a sigmoid function to convert distance to binding strength
        def distance_to_strength(distance, midpoint=4.0):
            if distance <= 0:  # No distance data available
                return 0
            return 1.0 / (1.0 + np.exp(distance - midpoint))
        
        # Calculate binding scores
        heavy_binding_scores = np.zeros(len(heavy_seq))
        light_binding_scores = np.zeros(len(light_seq))
        rbd_binding_scores = np.zeros(len(rbd_seq))
        
        # Heavy chain binding scores
        for i in range(len(heavy_seq)):
            for j in range(len(rbd_seq)):
                if heavy_rbd_contacts[i, j] > 0:
                    heavy_binding_scores[i] += distance_to_strength(heavy_rbd_contacts[i, j])
        
        # Light chain binding scores
        for i in range(len(light_seq)):
            for j in range(len(rbd_seq)):
                if light_rbd_contacts[i, j] > 0:
                    light_binding_scores[i] += distance_to_strength(light_rbd_contacts[i, j])
        
        # RBD binding scores
        for j in range(len(rbd_seq)):
            for i in range(len(heavy_seq)):
                if heavy_rbd_contacts[i, j] > 0:
                    rbd_binding_scores[j] += distance_to_strength(heavy_rbd_contacts[i, j])
            
            for i in range(len(light_seq)):
                if light_rbd_contacts[i, j] > 0:
                    rbd_binding_scores[j] += distance_to_strength(light_rbd_contacts[i, j])
        
        # Save binding scores
        np.save(os.path.join(DATA_DIR, 'heavy_binding_scores.npy'), heavy_binding_scores)
        np.save(os.path.join(DATA_DIR, 'light_binding_scores.npy'), light_binding_scores)
        np.save(os.path.join(DATA_DIR, 'rbd_binding_scores.npy'), rbd_binding_scores)
        
        # Save as CSV for easy importing
        pd.DataFrame({
            'residue': range(1, len(heavy_seq) + 1),
            'score': heavy_binding_scores
        }).to_csv(os.path.join(DATA_DIR, 'heavy_binding_scores.csv'), index=False)
        
        pd.DataFrame({
            'residue': range(1, len(light_seq) + 1),
            'score': light_binding_scores
        }).to_csv(os.path.join(DATA_DIR, 'light_binding_scores.csv'), index=False)
        
        pd.DataFrame({
            'residue': range(1, len(rbd_seq) + 1),
            'score': rbd_binding_scores
        }).to_csv(os.path.join(DATA_DIR, 'rbd_binding_scores.csv'), index=False)
        
        # Define CDR regions (Kabat numbering)
        heavy_cdrs = {
            'CDR-H1': range(26, 33),
            'CDR-H2': range(52, 57),
            'CDR-H3': range(95, 103)
        }

        light_cdrs = {
            'CDR-L1': range(24, 35),
            'CDR-L2': range(50, 57),
            'CDR-L3': range(89, 98)
        }
        
        # Analyze interface by CDR region
        heavy_interface_by_region = {}
        for cdr_name, cdr_range in heavy_cdrs.items():
            heavy_interface_by_region[cdr_name] = [res for res in heavy_interface if res in cdr_range]
        
        # Non-CDR residues
        all_heavy_cdrs = set()
        for cdr_range in heavy_cdrs.values():
            all_heavy_cdrs.update(cdr_range)
            
        heavy_interface_by_region['Framework'] = [res for res in heavy_interface if res not in all_heavy_cdrs]
        
        # Light chain interface by CDR
        light_interface_by_region = {}
        for cdr_name, cdr_range in light_cdrs.items():
            light_interface_by_region[cdr_name] = [res for res in light_interface if res in cdr_range]
        
        # Non-CDR residues
        all_light_cdrs = set()
        for cdr_range in light_cdrs.values():
            all_light_cdrs.update(cdr_range)
            
        light_interface_by_region['Framework'] = [res for res in light_interface if res not in all_light_cdrs]
        
        # Create a report on CDR distribution
        with open(os.path.join(DATA_DIR, 'cdr_distribution.txt'), 'w') as f:
            f.write("# CDR Distribution in Interface\n\n")
            
            # Heavy chain
            f.write("## Heavy Chain\n")
            for region, residues in heavy_interface_by_region.items():
                f.write(f"{region}: {len(residues)} residues - {residues}\n")
                
            # Count CDR vs framework residues
            heavy_cdr_count = sum(len(residues) for region, residues in heavy_interface_by_region.items() if region != 'Framework')
            heavy_framework_count = len(heavy_interface_by_region['Framework'])
            heavy_total = heavy_cdr_count + heavy_framework_count
            
            if heavy_total > 0:
                f.write(f"\nCDR residues: {heavy_cdr_count}/{heavy_total} ({heavy_cdr_count/heavy_total:.1%})\n")
                f.write(f"Framework residues: {heavy_framework_count}/{heavy_total} ({heavy_framework_count/heavy_total:.1%})\n\n")
            
            # Light chain
            f.write("## Light Chain\n")
            for region, residues in light_interface_by_region.items():
                f.write(f"{region}: {len(residues)} residues - {residues}\n")
                
            # Count CDR vs framework residues
            light_cdr_count = sum(len(residues) for region, residues in light_interface_by_region.items() if region != 'Framework')
            light_framework_count = len(light_interface_by_region['Framework'])
            light_total = light_cdr_count + light_framework_count
            
            if light_total > 0:
                f.write(f"\nCDR residues: {light_cdr_count}/{light_total} ({light_cdr_count/light_total:.1%})\n")
                f.write(f"Framework residues: {light_framework_count}/{light_total} ({light_framework_count/light_total:.1%})\n")
        
        # Create binary vectors for interface residues
        def make_binary_vector(residue_list, length):
            vector = np.zeros(length)
            for res in residue_list:
                if 0 <= res-1 < length:  # Adjust for 1-indexing
                    vector[res-1] = 1
            return vector
        
        heavy_interface_vec = make_binary_vector(heavy_interface, len(heavy_seq))
        light_interface_vec = make_binary_vector(light_interface, len(light_seq))
        rbd_interface_vec = make_binary_vector(rbd_interface, len(rbd_seq))
        
        return {
            'heavy_interface': heavy_interface,
            'light_interface': light_interface,
            'rbd_interface': rbd_interface,
            'heavy_binding_scores': heavy_binding_scores,
            'light_binding_scores': light_binding_scores,
            'rbd_binding_scores': rbd_binding_scores,
            'heavy_interface_by_region': heavy_interface_by_region,
            'light_interface_by_region': light_interface_by_region,
            'heavy_rbd_contacts': heavy_rbd_contacts,
            'light_rbd_contacts': light_rbd_contacts,
            'heavy_interface_vec': heavy_interface_vec,
            'light_interface_vec': light_interface_vec,
            'rbd_interface_vec': rbd_interface_vec,
            'heavy_cdrs': heavy_cdrs,
            'light_cdrs': light_cdrs
        }
    
    except Exception as e:
        print(f"Error analyzing interface: {e}")
        import traceback
        traceback.print_exc()
        return None

# Function to visualize interface analysis results
def visualize_results(interface_data):
    """Create visualizations of the interface analysis results"""
    print("Creating visualizations...")
    
    # Extract data from the interface analysis
    heavy_binding_scores = interface_data['heavy_binding_scores']
    light_binding_scores = interface_data['light_binding_scores']
    rbd_binding_scores = interface_data['rbd_binding_scores']
    heavy_rbd_contacts = interface_data['heavy_rbd_contacts']
    light_rbd_contacts = interface_data['light_rbd_contacts']
    heavy_interface_vec = interface_data['heavy_interface_vec']
    light_interface_vec = interface_data['light_interface_vec']
    rbd_interface_vec = interface_data['rbd_interface_vec']
    heavy_cdrs = interface_data['heavy_cdrs']
    light_cdrs = interface_data['light_cdrs']
    
    # 1. Create contact map heatmaps
    plt.figure(figsize=(12, 10))
    
    # Heavy chain - RBD contacts
    plt.subplot(2, 1, 1)
    # Use a copy of the contacts for visualization
    heavy_contacts_viz = heavy_rbd_contacts.copy()
    # Cap the distances at 12Å for better visualization
    heavy_contacts_viz[heavy_contacts_viz > 12] = 12
    # Convert distances to contact strength for visualization (closer = stronger)
    heavy_viz = 1 - (heavy_contacts_viz / 12)
    heavy_viz[heavy_rbd_contacts == 0] = 0  # No distance data = no contact
    
    sns.heatmap(heavy_viz, cmap='viridis')
    plt.title('Heavy Chain - RBD Contact Map')
    plt.xlabel('RBD Residue Position')
    plt.ylabel('Heavy Chain Residue Position')
    
    # Light chain - RBD contacts
    plt.subplot(2, 1, 2)
    light_contacts_viz = light_rbd_contacts.copy()
    light_contacts_viz[light_contacts_viz > 12] = 12
    light_viz = 1 - (light_contacts_viz / 12)
    light_viz[light_rbd_contacts == 0] = 0
    
    sns.heatmap(light_viz, cmap='viridis')
    plt.title('Light Chain - RBD Contact Map')
    plt.xlabel('RBD Residue Position')
    plt.ylabel('Light Chain Residue Position')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'contact_maps.png'), dpi=300)
    
    # 2. Plot binding scores with CDR regions highlighted
    plt.figure(figsize=(15, 12))
    
    plt.subplot(3, 1, 1)
    plt.bar(range(1, len(heavy_seq)+1), heavy_binding_scores)
    plt.title('Heavy Chain Binding Scores')
    plt.xlabel('Residue Position')
    plt.ylabel('Binding Score')
    
    # Highlight CDRs
    colors = ['lightgreen', 'lightblue', 'salmon']
    for i, (cdr_name, cdr_range) in enumerate(heavy_cdrs.items()):
        plt.axvspan(min(cdr_range), max(cdr_range), alpha=0.3, color=colors[i], label=cdr_name)
    
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.bar(range(1, len(light_seq)+1), light_binding_scores)
    plt.title('Light Chain Binding Scores')
    plt.xlabel('Residue Position')
    plt.ylabel('Binding Score')
    
    # Highlight CDRs
    for i, (cdr_name, cdr_range) in enumerate(light_cdrs.items()):
        plt.axvspan(min(cdr_range), max(cdr_range), alpha=0.3, color=colors[i], label=cdr_name)
    
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.bar(range(1, len(rbd_seq)+1), rbd_binding_scores)
    plt.title('RBD Binding Scores')
    plt.xlabel('Residue Position')
    plt.ylabel('Binding Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'binding_scores.png'), dpi=300)
    
    # 3. Plot interface residues with CDR regions
    plt.figure(figsize=(15, 10))
    
    # Heavy chain interface
    plt.subplot(2, 1, 1)
    plt.bar(range(1, len(heavy_seq)+1), heavy_interface_vec, color='blue')
    plt.title('Heavy Chain Interface Residues')
    plt.xlabel('Residue Position')
    plt.ylabel('Interface (1=yes, 0=no)')
    plt.ylim(-0.1, 1.1)
    
    # Highlight CDRs
    for i, (cdr_name, cdr_range) in enumerate(heavy_cdrs.items()):
        plt.axvspan(min(cdr_range), max(cdr_range), alpha=0.3, color=colors[i], label=cdr_name)
    
    plt.legend()
    
    # Light chain interface
    plt.subplot(2, 1, 2)
    plt.bar(range(1, len(light_seq)+1), light_interface_vec, color='red')
    plt.title('Light Chain Interface Residues')
    plt.xlabel('Residue Position')
    plt.ylabel('Interface (1=yes, 0=no)')
    plt.ylim(-0.1, 1.1)
    
    # Highlight CDRs
    for i, (cdr_name, cdr_range) in enumerate(light_cdrs.items()):
        plt.axvspan(min(cdr_range), max(cdr_range), alpha=0.3, color=colors[i], label=cdr_name)
    
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'interface_residues.png'), dpi=300)
    
    # 4. Plot RBD interface residues
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(rbd_seq)+1), rbd_interface_vec, color='green')
    plt.title('RBD Interface Residues')
    plt.xlabel('Residue Position')
    plt.ylabel('Interface (1=yes, 0=no)')
    plt.ylim(-0.1, 1.1)
    plt.savefig(os.path.join(PLOTS_DIR, 'rbd_interface.png'), dpi=300)
    
    # 5. Interface hotspots (using smoothed binding scores)
    plt.figure(figsize=(15, 15))
    
    # Smooth binding scores for hotspot detection
    heavy_smoothed = gaussian_filter1d(heavy_binding_scores, sigma=3)
    light_smoothed = gaussian_filter1d(light_binding_scores, sigma=3)
    rbd_smoothed = gaussian_filter1d(rbd_binding_scores, sigma=3)
    
    # Find local maxima
    heavy_peaks, _ = find_peaks(heavy_smoothed, height=np.percentile(heavy_smoothed, 90))
    light_peaks, _ = find_peaks(light_smoothed, height=np.percentile(light_smoothed, 90))
    rbd_peaks, _ = find_peaks(rbd_smoothed, height=np.percentile(rbd_smoothed, 90))
    
    # Convert to residue positions (1-indexed)
    heavy_hotspots = [i+1 for i in heavy_peaks]
    light_hotspots = [i+1 for i in light_peaks]
    rbd_hotspots = [i+1 for i in rbd_peaks]
    
    # Save hotspots
    with open(os.path.join(DATA_DIR, 'binding_hotspots.txt'), 'w') as f:
        f.write(f"Heavy chain binding hotspots: {heavy_hotspots}\n")
        f.write(f"Light chain binding hotspots: {light_hotspots}\n")
        f.write(f"RBD binding hotspots: {rbd_hotspots}\n")
    
    # Plot hotspots
    plt.subplot(3, 1, 1)
    plt.plot(range(1, len(heavy_seq)+1), heavy_smoothed, 'k-', alpha=0.7)
    plt.bar(range(1, len(heavy_seq)+1), heavy_binding_scores, alpha=0.5)
    plt.plot(heavy_hotspots, [heavy_binding_scores[i-1] for i in heavy_hotspots], 'ro', label='Hotspots')
    plt.title('Heavy Chain Binding Hotspots')
    plt.xlabel('Residue Position')
    plt.ylabel('Binding Score')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(range(1, len(light_seq)+1), light_smoothed, 'k-', alpha=0.7)
    plt.bar(range(1, len(light_seq)+1), light_binding_scores, alpha=0.5)
    plt.plot(light_hotspots, [light_binding_scores[i-1] for i in light_hotspots], 'ro', label='Hotspots')
    plt.title('Light Chain Binding Hotspots')
    plt.xlabel('Residue Position')
    plt.ylabel('Binding Score')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(range(1, len(rbd_seq)+1), rbd_smoothed, 'k-', alpha=0.7)
    plt.bar(range(1, len(rbd_seq)+1), rbd_binding_scores, alpha=0.5)
    plt.plot(rbd_hotspots, [rbd_binding_scores[i-1] for i in rbd_hotspots], 'ro', label='Hotspots')
    plt.title('RBD Binding Hotspots')
    plt.xlabel('Residue Position')
    plt.ylabel('Binding Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'binding_hotspots.png'), dpi=300)
    
    # 6. CDR contribution pie charts
    plt.figure(figsize=(12, 10))
    
    # Heavy chain interface distribution
    heavy_interface_by_region = interface_data['heavy_interface_by_region']
    heavy_cdr_count = sum(len(residues) for region, residues in heavy_interface_by_region.items() if region != 'Framework')
    heavy_framework_count = len(heavy_interface_by_region['Framework'])
    
    # Light chain interface distribution
    light_interface_by_region = interface_data['light_interface_by_region']
    light_cdr_count = sum(len(residues) for region, residues in light_interface_by_region.items() if region != 'Framework')
    light_framework_count = len(light_interface_by_region['Framework'])
    
    # Create pie charts
    plt.subplot(2, 1, 1)
    if heavy_cdr_count + heavy_framework_count > 0:
        plt.pie([heavy_cdr_count, heavy_framework_count], 
                labels=['CDRs', 'Framework'],
                autopct='%1.1f%%',
                colors=['lightblue', 'gray'])
        plt.title('Heavy Chain Interface Distribution')
    
    plt.subplot(2, 1, 2)
    if light_cdr_count + light_framework_count > 0:
        plt.pie([light_cdr_count, light_framework_count], 
                labels=['CDRs', 'Framework'],
                autopct='%1.1f%%',
                colors=['lightpink', 'gray'])
        plt.title('Light Chain Interface Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'cdr_distribution.png'), dpi=300)
    
    # 7. Detailed CDR breakdown
    plt.figure(figsize=(15, 10))
    
    # Heavy chain CDR breakdown
    plt.subplot(2, 1, 1)
    heavy_cdr_data = {k: len(v) for k, v in heavy_interface_by_region.items()}
    plt.bar(heavy_cdr_data.keys(), heavy_cdr_data.values())
    plt.title('Heavy Chain Interface by Region')
    plt.ylabel('Number of Interface Residues')
    plt.xticks(rotation=45)
    
    # Light chain CDR breakdown
    plt.subplot(2, 1, 2)
    light_cdr_data = {k: len(v) for k, v in light_interface_by_region.items()}
    plt.bar(light_cdr_data.keys(), light_cdr_data.values())
    plt.title('Light Chain Interface by Region')
    plt.ylabel('Number of Interface Residues')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'cdr_breakdown.png'), dpi=300)
    
    print("Visualizations complete")

def create_summary_report(interface_data):
    """Generate comprehensive summary report"""
    print("Creating summary report...")
    
    # Extract data
    heavy_interface = interface_data['heavy_interface']
    light_interface = interface_data['light_interface']
    rbd_interface = interface_data['rbd_interface']
    heavy_interface_by_region = interface_data['heavy_interface_by_region']
    light_interface_by_region = interface_data['light_interface_by_region']
    
    # Calculate CDR vs framework distribution
    heavy_cdr_count = sum(len(residues) for region, residues in heavy_interface_by_region.items() if region != 'Framework')
    heavy_framework_count = len(heavy_interface_by_region['Framework'])
    heavy_total = heavy_cdr_count + heavy_framework_count
    
    light_cdr_count = sum(len(residues) for region, residues in light_interface_by_region.items() if region != 'Framework')
    light_framework_count = len(light_interface_by_region['Framework'])
    light_total = light_cdr_count + light_framework_count
    
    # Create the summary report
    with open(os.path.join(RESULTS_DIR, 'summary_report.txt'), 'w') as f:
        f.write("# Structure-Based Interface Analysis\n\n")
        
        f.write("## Overview\n")
        f.write(f"Heavy chain sequence: {len(heavy_seq)} residues\n")
        f.write(f"Light chain sequence: {len(light_seq)} residues\n")
        f.write(f"RBD sequence: {len(rbd_seq)} residues\n\n")
        
        f.write("## Interface Statistics\n")
        f.write(f"Heavy chain interface residues: {len(heavy_interface)}\n")
        f.write(f"Light chain interface residues: {len(light_interface)}\n")
        f.write(f"RBD interface residues: {len(rbd_interface)}\n\n")
        
        if heavy_total > 0:
            f.write(f"Heavy chain CDR residues: {heavy_cdr_count} ({heavy_cdr_count/heavy_total:.1%})\n")
            f.write(f"Heavy chain framework residues: {heavy_framework_count} ({heavy_framework_count/heavy_total:.1%})\n")
        
        if light_total > 0:
            f.write(f"Light chain CDR residues: {light_cdr_count} ({light_cdr_count/light_total:.1%})\n")
            f.write(f"Light chain framework residues: {light_framework_count} ({light_framework_count/light_total:.1%})\n\n")
        
        f.write("## Interface by Region\n")
        for region, residues in heavy_interface_by_region.items():
            f.write(f"Heavy {region}: {len(residues)} residues\n")
            if residues:
                f.write(f"  Residues: {', '.join(map(str, residues))}\n")
        
        f.write("\n")
        for region, residues in light_interface_by_region.items():
            f.write(f"Light {region}: {len(residues)} residues\n")
            if residues:
                f.write(f"  Residues: {', '.join(map(str, residues))}\n")
        
        f.write("\n## Interface Residues\n")
        f.write(f"Heavy chain: {', '.join(map(str, heavy_interface))}\n\n")
        f.write(f"Light chain: {', '.join(map(str, light_interface))}\n\n")
        f.write(f"RBD: {', '.join(map(str, rbd_interface))}\n\n")
        
        f.write("\n## Additional Information\n")
        f.write("All data and visualizations are saved in the following directories:\n")
        f.write(f"- Raw data: {DATA_DIR}\n")
        f.write(f"- Plots: {PLOTS_DIR}\n")
        f.write(f"- Structures: {STRUCTURE_DIR}\n")
    
    print(f"Summary report saved to {os.path.join(RESULTS_DIR, 'summary_report.txt')}")

def main():
    """Run the complete structure-based analysis pipeline"""
    print("\nStarting structure-based interface analysis for CR3022-RBD complex\n" + "="*50)
    
    # Step 1: Analyze the interface from the structure
    interface_data = analyze_interface(structure_file)
    
    if interface_data:
        # Step 2: Visualize the results
        visualize_results(interface_data)
        
        # Step 3: Create summary report
        create_summary_report(interface_data)
        
        print("\nInterface analysis complete!\n")
        print(f"Results saved to: {RESULTS_DIR}")
        print(f"Summary report: {os.path.join(RESULTS_DIR, 'summary_report.txt')}")
    else:
        print("\nInterface analysis failed.")

if __name__ == "__main__":
    main()