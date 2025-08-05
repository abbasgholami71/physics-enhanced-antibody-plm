# scripts/prepare_sequences.py
from Bio import PDB
import os
from Bio.Data.IUPACData import protein_letters_3to1


# Create output directories
os.makedirs('structures', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Initialize parser
print("Parsing PDB structure...")
parser = PDB.PDBParser(QUIET=True)
structure = parser.get_structure('complex', 'structures/6W41.pdb')

# First, check available chains
print("Available chains in the structure:")
for chain in structure[0]:
    print(f"Chain {chain.id} with {len(list(chain.get_residues()))} residues")


# Define chains based on inspection
heavy_chain_id = 'H'  # Update this based on check_structure.py output
light_chain_id = 'L'  # Update this based on check_structure.py output
rbd_chain_id = 'C'    # Update this based on check_structure.py output

# Extract chains if they exist, otherwise notify
print("\nExtracting chains...")
io = PDB.PDBIO()

# Extract heavy chain
try:
    io.set_structure(structure[0][heavy_chain_id])
    io.save('structures/heavy_chain.pdb')
    print(f"Saved heavy chain (Chain {heavy_chain_id}) structure")
except KeyError:
    print(f"Error: Chain {heavy_chain_id} not found in structure!")

# Extract light chain
try:
    io.set_structure(structure[0][light_chain_id])
    io.save('structures/light_chain.pdb')
    print(f"Saved light chain (Chain {light_chain_id}) structure")
except KeyError:
    print(f"Error: Chain {light_chain_id} not found in structure!")

# Extract RBD
try:
    io.set_structure(structure[0][rbd_chain_id])
    io.save('structures/rbd.pdb')
    print(f"Saved RBD (Chain {rbd_chain_id}) structure")
except KeyError:
    print(f"Error: Chain {rbd_chain_id} not found in structure!")

# Extract Fab (chains H + L if both exist)
try:
    class ChainSelector(PDB.Select):
        def __init__(self, chain_letters):
            self.chain_letters = chain_letters
        
        def accept_chain(self, chain):
            return chain.id in self.chain_letters
    
    io.set_structure(structure)
    io.save('structures/fab.pdb', ChainSelector([heavy_chain_id, light_chain_id]))
    print(f"Saved Fab structure (Chains {heavy_chain_id} and {light_chain_id})")
except Exception as e:
    print(f"Error extracting Fab: {e}")

# Create a helper function at the top of your script
def three_to_one(three_letter_code):
    """Convert three letter amino acid code to one letter code."""
    # Standard amino acids are usually in all caps in PDB files, 
    # but the dictionary expects standard format (first letter capitalized)
    standard_format = three_letter_code.capitalize()
    return protein_letters_3to1.get(standard_format, 'X')

# Extract sequences
print("\nExtracting sequences...")

try:
    with open('data/heavy_chain.fasta', 'w') as f:
        f.write('>CR3022_heavy_chain\n')
        for residue in structure[0][heavy_chain_id]:
            if PDB.is_aa(residue):
                f.write(three_to_one(residue.get_resname()))
        f.write('\n')
    print("Saved heavy chain sequence")
except KeyError:
    print("Couldn't extract heavy chain sequence")

try:
    with open('data/light_chain.fasta', 'w') as f:
        f.write('>CR3022_light_chain\n')
        for residue in structure[0][light_chain_id]:
            if PDB.is_aa(residue):
                f.write(three_to_one(residue.get_resname()))
        f.write('\n')
    print("Saved light chain sequence")
except KeyError:
    print("Couldn't extract light chain sequence")

try:
    with open('data/rbd.fasta', 'w') as f:
        f.write('>SARS-CoV-2_RBD\n')
        for residue in structure[0][rbd_chain_id]:
            if PDB.is_aa(residue):
                f.write(three_to_one(residue.get_resname()))
        f.write('\n')
    print("Saved RBD sequence")
except KeyError:
    print("Couldn't extract RBD sequence")

# Create combined Fab sequence if both chains exist
try:
    with open('data/fab.fasta', 'w') as f:
        f.write('>CR3022_Fab\n')
        
        # Write heavy chain sequence
        for residue in structure[0][heavy_chain_id]:
            if PDB.is_aa(residue):
                f.write(three_to_one(residue.get_resname()))
        
        # Add separator
        f.write(':')
        
        # Write light chain sequence
        for residue in structure[0][light_chain_id]:
            if PDB.is_aa(residue):
                f.write(three_to_one(residue.get_resname()))
        
        f.write('\n')
    print("Saved Fab sequence")
except KeyError:
    print("Couldn't create combined Fab sequence")

# Create combined fasta for complex predictions
try:
    with open('data/complex.fasta', 'w') as f:
        f.write('>CR3022_Fab_heavy\n')
        for residue in structure[0][heavy_chain_id]:
            if PDB.is_aa(residue):
                f.write(three_to_one(residue.get_resname()))
        f.write('\n')
        
        f.write('>CR3022_Fab_light\n')
        for residue in structure[0][light_chain_id]:
            if PDB.is_aa(residue):
                f.write(three_to_one(residue.get_resname()))
        f.write('\n')
        
        f.write('>SARS-CoV-2_RBD\n')
        for residue in structure[0][rbd_chain_id]:
            if PDB.is_aa(residue):
                f.write(three_to_one(residue.get_resname()))
        f.write('\n')
    print("Created combined sequence file for complex")
except KeyError:
    print("Couldn't create complex sequence file")

print("\nPreprocessing completed!")