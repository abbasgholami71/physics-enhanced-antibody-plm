# scripts/download_structure.py
import os
from urllib import request

# Create directory for PDB files if it doesn't exist
os.makedirs('structures', exist_ok=True)

# Download PDB file
pdb_url = "https://files.rcsb.org/download/6W41.pdb"
pdb_file = "structures/6W41.pdb"

print(f"Downloading PDB structure from {pdb_url}...")
request.urlretrieve(pdb_url, pdb_file)
print(f"Downloaded successfully to {pdb_file}")

# Check file size to confirm successful download
file_size = os.path.getsize(pdb_file) / 1024  # Size in KB
print(f"File size: {file_size:.1f} KB")
