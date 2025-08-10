# scripts/esm2_analysis.py
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Create results directory
os.makedirs('results', exist_ok=True)
os.makedirs('results/esm2/plots', exist_ok=True)

print("Checking for GPU availability...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

# We'll use the transformers library as a fallback if fair-esm has issues
try:
    print("Trying to load ESM-2 model via fair-esm...")
    import esm
    
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    
    print("Preparing batch data...")
    data = [
        ("heavy_chain", heavy_seq),
        ("light_chain", light_seq),
        ("rbd", rbd_seq)
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    
    # Extract token representations
    print("Running inference with ESM-2...")
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])
    token_representations = results["representations"][33].cpu()
    
    # Get per-residue representations
    heavy_repr = token_representations[0, 1:len(heavy_seq)+1].numpy()
    light_repr = token_representations[1, 1:len(light_seq)+1].numpy()
    rbd_repr = token_representations[2, 1:len(rbd_seq)+1].numpy()
    
except Exception as e:
    print(f"Error loading via fair-esm: {e}")
    print("Falling back to transformers library...")
    
    from transformers import AutoTokenizer, AutoModel
    
    # Load ESM-2 model through transformers
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)
    model.eval()
    
    # Process sequences
    def get_embeddings(sequence):
        inputs = tokenizer(sequence, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get per-residue embeddings (remove special tokens)
        embeddings = outputs.last_hidden_state[0, 1:len(sequence)+1].cpu().numpy()
        return embeddings
    
    print("Running inference for heavy chain...")
    heavy_repr = get_embeddings(heavy_seq)
    
    print("Running inference for light chain...")
    light_repr = get_embeddings(light_seq)
    
    print("Running inference for RBD...")
    rbd_repr = get_embeddings(rbd_seq)

print(f"Generated embeddings with shape: heavy {heavy_repr.shape}, light {light_repr.shape}, RBD {rbd_repr.shape}")

# Save embeddings
np.save('results/esm2/heavy_chain_embeddings.npy', heavy_repr)
np.save('results/esm2/light_chain_embeddings.npy', light_repr)
np.save('results/esm2/rbd_embeddings.npy', rbd_repr)
print("Saved embeddings to results directory")

# Calculate similarity matrices for binding prediction
print("Calculating binding similarity matrices...")

# Heavy chain vs RBD
heavy_rbd_sim = np.zeros((len(heavy_seq), len(rbd_seq)))
for i in range(len(heavy_seq)):
    for j in range(len(rbd_seq)):
        heavy_rbd_sim[i, j] = np.dot(heavy_repr[i], rbd_repr[j]) / (np.linalg.norm(heavy_repr[i]) * np.linalg.norm(rbd_repr[j]))

# Light chain vs RBD
light_rbd_sim = np.zeros((len(light_seq), len(rbd_seq)))
for i in range(len(light_seq)):
    for j in range(len(rbd_seq)):
        light_rbd_sim[i, j] = np.dot(light_repr[i], rbd_repr[j]) / (np.linalg.norm(light_repr[i]) * np.linalg.norm(rbd_repr[j]))

np.save('results/esm2/heavy_rbd_similarity.npy', heavy_rbd_sim)
np.save('results/esm2/light_rbd_similarity.npy', light_rbd_sim)

# Visualize the similarity matrices
plt.figure(figsize=(12, 10))
plt.subplot(2, 1, 1)
sns.heatmap(heavy_rbd_sim, cmap='viridis')
plt.title('Heavy Chain-RBD Residue Similarity Matrix')
plt.xlabel('RBD Residue Position')
plt.ylabel('Heavy Chain Residue Position')

plt.subplot(2, 1, 2)
sns.heatmap(light_rbd_sim, cmap='viridis')
plt.title('Light Chain-RBD Residue Similarity Matrix')
plt.xlabel('RBD Residue Position')
plt.ylabel('Light Chain Residue Position')

plt.tight_layout()
plt.savefig('results/esm2/plots/binding_similarity_heatmaps.png', dpi=300, bbox_inches='tight')
print("Saved similarity heatmaps")

# Predict binding residues (top 20% of similarity scores)
print("Predicting binding residues...")

# Heavy chain binding prediction
heavy_binding_scores = np.max(heavy_rbd_sim, axis=1)
heavy_binding_threshold = np.percentile(heavy_binding_scores, 80)
heavy_binding_residues = [i+1 for i in range(len(heavy_seq)) if heavy_binding_scores[i] > heavy_binding_threshold]

# Light chain binding prediction
light_binding_scores = np.max(light_rbd_sim, axis=1)
light_binding_threshold = np.percentile(light_binding_scores, 80)
light_binding_residues = [i+1 for i in range(len(light_seq)) if light_binding_scores[i] > light_binding_threshold]

# RBD binding prediction (combine scores from heavy and light)
rbd_binding_scores_heavy = np.max(heavy_rbd_sim, axis=0)
rbd_binding_scores_light = np.max(light_rbd_sim, axis=0)
rbd_binding_scores = np.maximum(rbd_binding_scores_heavy, rbd_binding_scores_light)
rbd_binding_threshold = np.percentile(rbd_binding_scores, 80)
rbd_binding_residues = [i+1 for i in range(len(rbd_seq)) if rbd_binding_scores[i] > rbd_binding_threshold]

# Save results
with open('results/esm2/plm_binding_predictions.txt', 'w') as f:
    f.write(f"Heavy chain binding residues: {heavy_binding_residues}\n")
    f.write(f"Light chain binding residues: {light_binding_residues}\n")
    f.write(f"RBD binding residues: {rbd_binding_residues}\n")

# Plot binding scores
plt.figure(figsize=(15, 12))
plt.subplot(3, 1, 1)
plt.bar(range(1, len(heavy_seq)+1), heavy_binding_scores)
plt.axhline(y=heavy_binding_threshold, color='r', linestyle='--')
plt.title('Heavy Chain Binding Scores')
plt.xlabel('Residue Position')
plt.ylabel('Binding Score')

plt.subplot(3, 1, 2)
plt.bar(range(1, len(light_seq)+1), light_binding_scores)
plt.axhline(y=light_binding_threshold, color='r', linestyle='--')
plt.title('Light Chain Binding Scores')
plt.xlabel('Residue Position')
plt.ylabel('Binding Score')

plt.subplot(3, 1, 3)
plt.bar(range(1, len(rbd_seq)+1), rbd_binding_scores)
plt.axhline(y=rbd_binding_threshold, color='r', linestyle='--')
plt.title('RBD Binding Scores')
plt.xlabel('Residue Position')
plt.ylabel('Binding Score')

plt.tight_layout()
plt.savefig('results/esm2/plots/binding_scores.png', dpi=300, bbox_inches='tight')

# Create approximate CDR regions based on typical Kabat numbering
# Note: This is approximate since we don't have the exact numbering in our fasta sequences
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

plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.bar(range(1, len(heavy_seq)+1), heavy_binding_scores)
plt.axhline(y=heavy_binding_threshold, color='r', linestyle='--')

# Highlight CDRs
colors = ['lightgreen', 'lightblue', 'salmon']
for i, (cdr_name, cdr_range) in enumerate(heavy_cdrs.items()):
    plt.axvspan(min(cdr_range), max(cdr_range), alpha=0.3, color=colors[i], label=cdr_name)

plt.title('Heavy Chain Binding Scores with Approximate CDRs')
plt.xlabel('Residue Position')
plt.ylabel('Binding Score')
plt.legend()

plt.subplot(2, 1, 2)
plt.bar(range(1, len(light_seq)+1), light_binding_scores)
plt.axhline(y=light_binding_threshold, color='r', linestyle='--')

# Highlight CDRs
for i, (cdr_name, cdr_range) in enumerate(light_cdrs.items()):
    plt.axvspan(min(cdr_range), max(cdr_range), alpha=0.3, color=colors[i], label=cdr_name)

plt.title('Light Chain Binding Scores with Approximate CDRs')
plt.xlabel('Residue Position')
plt.ylabel('Binding Score')
plt.legend()

plt.tight_layout()
plt.savefig('results/esm2/plots/binding_scores_with_cdrs.png', dpi=300, bbox_inches='tight')

print("Analysis complete! Results saved to the results directory.")

print("\nPerforming additional binding analyses...")

# 1. Calculate pseudo-interaction energy
print("Calculating interaction energy approximation...")
def calc_interaction_energy(chain_repr, rbd_repr):
    energy_matrix = np.zeros((len(chain_repr), len(rbd_repr)))
    for i in range(len(chain_repr)):
        for j in range(len(rbd_repr)):
            # Convert similarity to distance-like metric
            energy_matrix[i, j] = -np.log(0.1 + chain_repr[i, j])
    return energy_matrix

heavy_rbd_energy = calc_interaction_energy(heavy_rbd_sim, rbd_repr)
light_rbd_energy = calc_interaction_energy(light_rbd_sim, rbd_repr)

# Save energy matrices
np.save('results/esm2/heavy_rbd_energy.npy', heavy_rbd_energy)
np.save('results/esm2/light_rbd_energy.npy', light_rbd_energy)

# 2. Identify potential interface residues
print("Identifying interface residues...")
heavy_interface = [i+1 for i in range(len(heavy_seq)) 
                  if np.any(heavy_rbd_sim[i] > 0.7)]
light_interface = [i+1 for i in range(len(light_seq)) 
                  if np.any(light_rbd_sim[i] > 0.7)]
rbd_interface = [i+1 for i in range(len(rbd_seq)) 
                if np.max([np.max(heavy_rbd_sim[:, i]), np.max(light_rbd_sim[:, i])]) > 0.7]

# Save interface residues
with open('results/esm2/interface_residues.txt', 'w') as f:
    f.write(f"Heavy chain interface residues: {heavy_interface}\n")
    f.write(f"Light chain interface residues: {light_interface}\n")
    f.write(f"RBD interface residues: {rbd_interface}\n")

# 3. Binding Hotspot Identification
print("Finding binding hotspots...")
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# Smooth scores to identify regions
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
with open('results/esm2/binding_hotspots.txt', 'w') as f:
    f.write(f"Heavy chain binding hotspots: {heavy_hotspots}\n")
    f.write(f"Light chain binding hotspots: {light_hotspots}\n")
    f.write(f"RBD binding hotspots: {rbd_hotspots}\n")

# Plot the hotspots with smoothed curves
plt.figure(figsize=(15, 12))
plt.subplot(3, 1, 1)
plt.plot(range(1, len(heavy_seq)+1), heavy_smoothed, 'k-', alpha=0.7)
plt.bar(range(1, len(heavy_seq)+1), heavy_binding_scores, alpha=0.5)
plt.plot(heavy_hotspots, [heavy_binding_scores[i-1] for i in heavy_hotspots], 'ro', label='Hotspots')
plt.axhline(y=heavy_binding_threshold, color='r', linestyle='--')
plt.title('Heavy Chain Binding Hotspots')
plt.xlabel('Residue Position')
plt.ylabel('Binding Score')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(range(1, len(light_seq)+1), light_smoothed, 'k-', alpha=0.7)
plt.bar(range(1, len(light_seq)+1), light_binding_scores, alpha=0.5)
plt.plot(light_hotspots, [light_binding_scores[i-1] for i in light_hotspots], 'ro', label='Hotspots')
plt.axhline(y=light_binding_threshold, color='r', linestyle='--')
plt.title('Light Chain Binding Hotspots')
plt.xlabel('Residue Position')
plt.ylabel('Binding Score')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(range(1, len(rbd_seq)+1), rbd_smoothed, 'k-', alpha=0.7)
plt.bar(range(1, len(rbd_seq)+1), rbd_binding_scores, alpha=0.5)
plt.plot(rbd_hotspots, [rbd_binding_scores[i-1] for i in rbd_hotspots], 'ro', label='Hotspots')
plt.axhline(y=rbd_binding_threshold, color='r', linestyle='--')
plt.title('RBD Binding Hotspots')
plt.xlabel('Residue Position')
plt.ylabel('Binding Score')
plt.legend()

plt.tight_layout()
plt.savefig('results/esm2/plots/binding_hotspots.png', dpi=300, bbox_inches='tight')

# 4. Epitope-Paratope Mapping
print("Creating epitope-paratope mapping...")
epitope_paratope_map = {}
epitope_paratope_scores = {}

for j in range(len(rbd_seq)):
    # Find max similarity score from either chain
    max_heavy_idx = np.argmax(heavy_rbd_sim[:, j])
    max_light_idx = np.argmax(light_rbd_sim[:, j])
    
    max_heavy_score = heavy_rbd_sim[max_heavy_idx, j]
    max_light_score = light_rbd_sim[max_light_idx, j]
    
    if max_heavy_score > max_light_score:
        epitope_paratope_map[f"RBD_{j+1}"] = f"H_{max_heavy_idx+1}"
        epitope_paratope_scores[f"RBD_{j+1}"] = max_heavy_score
    else:
        epitope_paratope_map[f"RBD_{j+1}"] = f"L_{max_light_idx+1}"
        epitope_paratope_scores[f"RBD_{j+1}"] = max_light_score

# Save epitope-paratope mapping
with open('results/esm2/epitope_paratope_map.txt', 'w') as f:
    f.write("RBD_Residue\tAntibody_Residue\tSimilarity_Score\n")
    for rbd_res, ab_res in epitope_paratope_map.items():
        f.write(f"{rbd_res}\t{ab_res}\t{epitope_paratope_scores[rbd_res]:.4f}\n")

# 5. Top interactions table
print("Identifying top interaction pairs...")
# Flatten and sort all interaction scores
all_interactions = []
for i in range(len(heavy_seq)):
    for j in range(len(rbd_seq)):
        all_interactions.append((f"H_{i+1}", f"RBD_{j+1}", heavy_rbd_sim[i, j]))

for i in range(len(light_seq)):
    for j in range(len(rbd_seq)):
        all_interactions.append((f"L_{i+1}", f"RBD_{j+1}", light_rbd_sim[i, j]))

# Sort by score (descending)
all_interactions.sort(key=lambda x: x[2], reverse=True)

# Save top 100 interactions
with open('results/esm2/top_interactions.txt', 'w') as f:
    f.write("Antibody_Residue\tRBD_Residue\tSimilarity_Score\n")
    for ab_res, rbd_res, score in all_interactions[:100]:
        f.write(f"{ab_res}\t{rbd_res}\t{score:.4f}\n")

# 6. Total binding energy per chain
print("Calculating binding energy totals...")
heavy_total_energy = np.sum(np.max(heavy_rbd_sim, axis=1))
light_total_energy = np.sum(np.max(light_rbd_sim, axis=1))

with open('results/esm2/binding_energy_summary.txt', 'w') as f:
    f.write(f"Heavy chain total binding energy: {heavy_total_energy:.4f}\n")
    f.write(f"Light chain total binding energy: {light_total_energy:.4f}\n")
    f.write(f"Ratio (Heavy/Light): {heavy_total_energy/light_total_energy:.4f}\n")

print("Additional analyses complete! All results saved to the results directory.")