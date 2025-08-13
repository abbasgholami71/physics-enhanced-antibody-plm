# scripts/antiberty+esm2_analysis.py
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.decomposition import PCA

# Create results directory
os.makedirs('results/antiberty-esm2/plots', exist_ok=True)

#------------------------------------------------------------------------------
# Setup and Data Loading
#------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------
# Model Loading (AntiBERTy for Antibodies)
#------------------------------------------------------------------------------
print("Loading AntiBERTy model...")
try:
    from antiberty import AntiBERTyRunner
    
    ab_model = AntiBERTyRunner()
    model = ab_model.model.to(device)
    model.eval()
    print("Successfully loaded AntiBERTy model")
    
    def get_antiberty_embeddings(sequence):
        """Get embeddings for a protein sequence using AntiBERTy"""
        embeddings = ab_model.embed([sequence])[0]
        # Remove special tokens (first and last token)
        return embeddings[1:-1].cpu().numpy()
        
except Exception as e:
    print(f"Error loading AntiBERTy: {e}")
    exit(1)

#------------------------------------------------------------------------------
# Model Loading (ESM-2 for RBD)
#------------------------------------------------------------------------------
print("Loading ESM-2 model for RBD...")
import esm
model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model_esm = model_esm.to(device)
model_esm.eval()

def get_esm2_embeddings(sequence):
    """Get embeddings for a protein sequence using ESM-2"""
    batch_converter = alphabet.get_batch_converter()
    data = [("protein", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    
    with torch.no_grad():
        results = model_esm(batch_tokens, repr_layers=[33])
    
    # Extract per-residue embeddings (excluding start/end tokens)
    embeddings = results["representations"][33][0, 1:len(sequence)+1].cpu().numpy()
    return embeddings

#------------------------------------------------------------------------------
# Generate Embeddings
#------------------------------------------------------------------------------
print("Running AntiBERTy inference for antibodies...")
heavy_repr = get_antiberty_embeddings(heavy_seq)
light_repr = get_antiberty_embeddings(light_seq)

print("Running ESM-2 inference for RBD...")
rbd_repr = get_esm2_embeddings(rbd_seq)

print(f"Generated embeddings with shape: heavy {heavy_repr.shape}, light {light_repr.shape}, RBD {rbd_repr.shape}")

# Save embeddings
np.save('results/antiberty-esm2/heavy_chain_embeddings.npy', heavy_repr)
np.save('results/antiberty-esm2/light_chain_embeddings.npy', light_repr)
np.save('results/antiberty-esm2/rbd_embeddings.npy', rbd_repr)
print("Saved embeddings to results directory")

#------------------------------------------------------------------------------
# Harmonize Embedding Dimensions
#------------------------------------------------------------------------------
print("Harmonizing embedding dimensions...")
# Calculate maximum possible components
max_components = min(
    min(heavy_repr.shape[0], heavy_repr.shape[1]),
    min(rbd_repr.shape[0], rbd_repr.shape[1])
)

# Set target dimension
target_dim = min(100, max_components)
print(f"Using {target_dim} components for dimension reduction")

# Fit PCA on antibody embeddings
pca_antiberty = PCA(n_components=target_dim)
heavy_pca = pca_antiberty.fit_transform(heavy_repr)
light_pca = pca_antiberty.transform(light_repr)

# Fit PCA on RBD embeddings
pca_esm = PCA(n_components=target_dim)
rbd_pca = pca_esm.fit_transform(rbd_repr)

print(f"Transformed dimensions: heavy {heavy_pca.shape}, light {light_pca.shape}, RBD {rbd_pca.shape}")
print(f"Variance explained by antibody PCA: {sum(pca_antiberty.explained_variance_ratio_):.2%}")
print(f"Variance explained by RBD PCA: {sum(pca_esm.explained_variance_ratio_):.2%}")

# Use the PCA projections for similarity calculations
heavy_repr, light_repr, rbd_repr = heavy_pca, light_pca, rbd_pca

# Save the hybrid embeddings
np.save('results/antiberty-esm2/heavy_chain_hybrid_embeddings.npy', heavy_repr)
np.save('results/antiberty-esm2/light_chain_hybrid_embeddings.npy', light_repr)
np.save('results/antiberty-esm2/rbd_hybrid_embeddings.npy', rbd_repr)
print("Saved hybrid embeddings")

#------------------------------------------------------------------------------
# Calculate Similarity Matrices
#------------------------------------------------------------------------------
print("Calculating binding similarity matrices...")

# Helper function for cosine similarity
def cosine_similarity_matrix(seq_a_emb, seq_b_emb):
    """Calculate cosine similarity between all pairs of embeddings"""
    sim_matrix = np.zeros((len(seq_a_emb), len(seq_b_emb)))
    for i in range(len(seq_a_emb)):
        for j in range(len(seq_b_emb)):
            sim_matrix[i, j] = np.dot(seq_a_emb[i], seq_b_emb[j]) / (
                np.linalg.norm(seq_a_emb[i]) * np.linalg.norm(seq_b_emb[j])
            )
    return sim_matrix

# Heavy chain vs RBD
heavy_rbd_sim = cosine_similarity_matrix(heavy_repr, rbd_repr)

# Light chain vs RBD
light_rbd_sim = cosine_similarity_matrix(light_repr, rbd_repr)

np.save('results/antiberty-esm2/heavy_rbd_hybrid_similarity.npy', heavy_rbd_sim)
np.save('results/antiberty-esm2/light_rbd_hybrid_similarity.npy', light_rbd_sim)

#------------------------------------------------------------------------------
# Visualize Similarity Matrices
#------------------------------------------------------------------------------
plt.figure(figsize=(12, 10))
plt.subplot(2, 1, 1)
sns.heatmap(heavy_rbd_sim, cmap='viridis')
plt.title('Heavy Chain-RBD Residue Similarity Matrix (Hybrid)')
plt.xlabel('RBD Residue Position')
plt.ylabel('Heavy Chain Residue Position')

plt.subplot(2, 1, 2)
sns.heatmap(light_rbd_sim, cmap='viridis')
plt.title('Light Chain-RBD Residue Similarity Matrix (Hybrid)')
plt.xlabel('RBD Residue Position')
plt.ylabel('Light Chain Residue Position')

plt.tight_layout()
plt.savefig('results/antiberty-esm2/plots/hybrid_binding_similarity_heatmaps.png', dpi=300, bbox_inches='tight')
print("Saved similarity heatmaps")

#------------------------------------------------------------------------------
# Calculate Binding Scores and Predict Binding Residues
#------------------------------------------------------------------------------
print("Predicting binding residues...")

# Extract binding scores (maximum similarity across RBD residues)
heavy_binding_scores = np.max(heavy_rbd_sim, axis=1)
light_binding_scores = np.max(light_rbd_sim, axis=1)
rbd_binding_scores_heavy = np.max(heavy_rbd_sim, axis=0)
rbd_binding_scores_light = np.max(light_rbd_sim, axis=0)
rbd_binding_scores = np.maximum(rbd_binding_scores_heavy, rbd_binding_scores_light)

# Calculate thresholds (top 20%)
heavy_binding_threshold = np.percentile(heavy_binding_scores, 80)
light_binding_threshold = np.percentile(light_binding_scores, 80)
rbd_binding_threshold = np.percentile(rbd_binding_scores, 80)

# Identify binding residues (above threshold)
heavy_binding_residues = [i+1 for i in range(len(heavy_seq)) if heavy_binding_scores[i] > heavy_binding_threshold]
light_binding_residues = [i+1 for i in range(len(light_seq)) if light_binding_scores[i] > light_binding_threshold]
rbd_binding_residues = [i+1 for i in range(len(rbd_seq)) if rbd_binding_scores[i] > rbd_binding_threshold]

# Save results
with open('results/antiberty-esm2/binding_predictions.txt', 'w') as f:
    f.write(f"Heavy chain binding residues: {heavy_binding_residues}\n")
    f.write(f"Light chain binding residues: {light_binding_residues}\n")
    f.write(f"RBD binding residues: {rbd_binding_residues}\n")

#------------------------------------------------------------------------------
# Define CDR Regions
#------------------------------------------------------------------------------
# Define approximate CDR regions based on typical Kabat numbering
heavy_cdrs = {
    'CDR-H1': range(26, 33),
    'CDR-H2': range(52, 58), 
    'CDR-H3': range(95, 103)
}

light_cdrs = {
    'CDR-L1': range(24, 35),
    'CDR-L2': range(50, 57),
    'CDR-L3': range(89, 98)
}

#------------------------------------------------------------------------------
# Plot Binding Scores with CDR Regions
#------------------------------------------------------------------------------
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.bar(range(1, len(heavy_seq)+1), heavy_binding_scores)
plt.axhline(y=heavy_binding_threshold, color='r', linestyle='--')

# Highlight CDRs
colors = ['lightgreen', 'lightblue', 'salmon']
for i, (cdr_name, cdr_range) in enumerate(heavy_cdrs.items()):
    plt.axvspan(min(cdr_range), max(cdr_range), alpha=0.3, color=colors[i], label=cdr_name)

plt.title('Heavy Chain Binding Scores with CDRs (Hybrid)')
plt.xlabel('Residue Position')
plt.ylabel('Binding Score')
plt.legend()

plt.subplot(2, 1, 2)
plt.bar(range(1, len(light_seq)+1), light_binding_scores)
plt.axhline(y=light_binding_threshold, color='r', linestyle='--')

# Highlight CDRs
for i, (cdr_name, cdr_range) in enumerate(light_cdrs.items()):
    plt.axvspan(min(cdr_range), max(cdr_range), alpha=0.3, color=colors[i], label=cdr_name)

plt.title('Light Chain Binding Scores with CDRs (Hybrid)')
plt.xlabel('Residue Position')
plt.ylabel('Binding Score')
plt.legend()

plt.tight_layout()
plt.savefig('results/antiberty-esm2/plots/binding_scores_with_cdrs.png', dpi=300, bbox_inches='tight')

#------------------------------------------------------------------------------
# Advanced Binding Analysis
#------------------------------------------------------------------------------
print("\nPerforming additional binding analyses...")

# 1. Calculate pseudo-interaction energy
print("Calculating interaction energy approximation...")
def calc_interaction_energy(sim_matrix):
    """Convert similarity to distance-like metric"""
    return -np.log(0.1 + sim_matrix)

heavy_rbd_energy = calc_interaction_energy(heavy_rbd_sim)
light_rbd_energy = calc_interaction_energy(light_rbd_sim)

# Save energy matrices
np.save('results/antiberty-esm2/heavy_rbd_energy.npy', heavy_rbd_energy)
np.save('results/antiberty-esm2/light_rbd_energy.npy', light_rbd_energy)

# 2. Identify potential interface residues
print("Identifying interface residues...")
INTERFACE_THRESHOLD = 0.7
heavy_interface = [i+1 for i in range(len(heavy_seq)) 
                  if np.any(heavy_rbd_sim[i] > INTERFACE_THRESHOLD)]
light_interface = [i+1 for i in range(len(light_seq)) 
                  if np.any(light_rbd_sim[i] > INTERFACE_THRESHOLD)]
rbd_interface = [i+1 for i in range(len(rbd_seq)) 
                if np.max([np.max(heavy_rbd_sim[:, i]), np.max(light_rbd_sim[:, i])]) > INTERFACE_THRESHOLD]

# Save interface residues
with open('results/antiberty-esm2/interface_residues.txt', 'w') as f:
    f.write(f"Heavy chain interface residues: {heavy_interface}\n")
    f.write(f"Light chain interface residues: {light_interface}\n")
    f.write(f"RBD interface residues: {rbd_interface}\n")

# 3. Binding Hotspot Identification
print("Finding binding hotspots...")
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
with open('results/antiberty-esm2/binding_hotspots.txt', 'w') as f:
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
plt.title('Heavy Chain Binding Hotspots (Hybrid)')
plt.xlabel('Residue Position')
plt.ylabel('Binding Score')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(range(1, len(light_seq)+1), light_smoothed, 'k-', alpha=0.7)
plt.bar(range(1, len(light_seq)+1), light_binding_scores, alpha=0.5)
plt.plot(light_hotspots, [light_binding_scores[i-1] for i in light_hotspots], 'ro', label='Hotspots')
plt.axhline(y=light_binding_threshold, color='r', linestyle='--')
plt.title('Light Chain Binding Hotspots (Hybrid)')
plt.xlabel('Residue Position')
plt.ylabel('Binding Score')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(range(1, len(rbd_seq)+1), rbd_smoothed, 'k-', alpha=0.7)
plt.bar(range(1, len(rbd_seq)+1), rbd_binding_scores, alpha=0.5)
plt.plot(rbd_hotspots, [rbd_binding_scores[i-1] for i in rbd_hotspots], 'ro', label='Hotspots')
plt.axhline(y=rbd_binding_threshold, color='r', linestyle='--')
plt.title('RBD Binding Hotspots (Hybrid)')
plt.xlabel('Residue Position')
plt.ylabel('Binding Score')
plt.legend()

plt.tight_layout()
plt.savefig('results/antiberty-esm2/plots/binding_hotspots.png', dpi=300, bbox_inches='tight')

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
with open('results/antiberty-esm2/epitope_paratope_map.txt', 'w') as f:
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
with open('results/antiberty-esm2/top_interactions.txt', 'w') as f:
    f.write("Antibody_Residue\tRBD_Residue\tSimilarity_Score\n")
    for ab_res, rbd_res, score in all_interactions[:100]:
        f.write(f"{ab_res}\t{rbd_res}\t{score:.4f}\n")

# 6. Total binding energy per chain
print("Calculating binding energy totals...")
heavy_total_energy = np.sum(np.max(heavy_rbd_sim, axis=1))
light_total_energy = np.sum(np.max(light_rbd_sim, axis=1))

with open('results/antiberty-esm2/binding_energy_summary.txt', 'w') as f:
    f.write(f"Heavy chain total binding energy: {heavy_total_energy:.4f}\n")
    f.write(f"Light chain total binding energy: {light_total_energy:.4f}\n")
    f.write(f"Ratio (Heavy/Light): {heavy_total_energy/light_total_energy:.4f}\n")

print("Analysis complete! Results saved to results/antiberty-esm2 directory.")