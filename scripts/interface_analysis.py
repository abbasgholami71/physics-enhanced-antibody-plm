#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Antibody-Antigen Interface Analysis

Analyzes the structural interface between an antibody and its antigen from a PDB file,
calculating key binding characteristics according to standard structural biology metrics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.PDB import *
from Bio.PDB import is_aa
from Bio.Data.IUPACData import protein_letters_3to1
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')


def three_to_one(three_letter_code):
    """Convert three letter amino acid code to one letter code."""
    standard_format = three_letter_code.capitalize()
    return protein_letters_3to1.get(standard_format, 'X')


class AntibodyInterface:
    def __init__(self, pdb_file, heavy_chain='H', light_chain='L', antigen_chain='C', 
                 output_dir='results/interface_analysis'):
        """Initialize with PDB file and chain identifiers"""
        self.pdb_file = pdb_file
        self.heavy_id = heavy_chain
        self.light_id = light_chain
        self.antigen_id = antigen_chain
        self.output_dir = output_dir
        self.interface_cutoff = 4.5  # Standard for interface definition (Angstroms)
        self.contact_cutoff = 8.0    # For broader contact analysis (Angstroms)
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        os.makedirs(f"{output_dir}/data", exist_ok=True)
        
        # Load structure
        print(f"Loading PDB structure: {pdb_file}")
        parser = PDBParser(QUIET=True)
        try:
            self.structure = parser.get_structure("complex", pdb_file)
            self.model = self.structure[0]  # First model
            
            # Extract chains
            self.heavy_chain = self.model[self.heavy_id]
            self.light_chain = self.model[self.light_id]
            self.antigen_chain = self.model[self.antigen_id]
            
            print(f"Successfully loaded chains: Heavy={self.heavy_id}, Light={self.light_id}, Antigen={self.antigen_id}")
        except Exception as e:
            print(f"Error loading structure: {e}")
            raise
            
        # Define CDR regions (Kabat numbering)
        self.heavy_cdrs = {
            'CDR-H1': range(31, 36),
            'CDR-H2': range(50, 66),
            'CDR-H3': range(95, 103)
        }
        
        self.light_cdrs = {
            'CDR-L1': range(24, 35),
            'CDR-L2': range(50, 57),
            'CDR-L3': range(89, 98)
        }

    def extract_sequences(self):
        """Extract sequences and residue information"""
        print("Extracting sequence information...")
        
        # Get standard residues (exclude hetero and water)
        self.heavy_residues = [r for r in self.heavy_chain.get_residues() if r.id[0] == ' ']
        self.light_residues = [r for r in self.light_chain.get_residues() if r.id[0] == ' ']
        self.antigen_residues = [r for r in self.antigen_chain.get_residues() if r.id[0] == ' ']
        
        # Extract sequences
        self.heavy_seq = ''.join([three_to_one(r.get_resname()) 
                                for r in self.heavy_residues if is_aa(r)])
        self.light_seq = ''.join([three_to_one(r.get_resname()) 
                                for r in self.light_residues if is_aa(r)])
        self.antigen_seq = ''.join([three_to_one(r.get_resname()) 
                                  for r in self.antigen_residues if is_aa(r)])
        
        # Store residue numbers for mapping
        self.heavy_res_ids = [(r.id[1], r.id[2]) for r in self.heavy_residues if is_aa(r)]
        self.light_res_ids = [(r.id[1], r.id[2]) for r in self.light_residues if is_aa(r)]
        self.antigen_res_ids = [(r.id[1], r.id[2]) for r in self.antigen_residues if is_aa(r)]
        
        # Save sequences to FASTA
        with open(f"{self.output_dir}/data/sequences.fasta", 'w') as f:
            f.write(f">Heavy_Chain\n{self.heavy_seq}\n")
            f.write(f">Light_Chain\n{self.light_seq}\n")
            f.write(f">Antigen\n{self.antigen_seq}\n")
        
        print(f"Heavy chain: {len(self.heavy_seq)} residues")
        print(f"Light chain: {len(self.light_seq)} residues")
        print(f"Antigen: {len(self.antigen_seq)} residues")
        
        return self.heavy_seq, self.light_seq, self.antigen_seq
    
    def _min_distance(self, res1, res2):
        """Calculate minimum distance between atoms of two residues"""
        min_dist = float('inf')
        atoms_found = False
        
        for atom1 in res1:
            for atom2 in res2:
                # Skip hydrogen atoms
                if atom1.element == 'H' or atom2.element == 'H':
                    continue
                
                atoms_found = True
                dist = atom1 - atom2
                if dist < min_dist:
                    min_dist = dist
                    
        return min_dist if atoms_found else float('inf')

    def identify_interface(self):
        """Identify interface residues using distance cutoff"""
        print(f"Calculating interface residues (cutoff: {self.interface_cutoff}Å)...")
        
        self.heavy_interface = set()
        self.light_interface = set()
        self.antigen_interface = set()
        
        # Calculate interface between heavy chain and antigen
        for h_res in self.heavy_residues:
            for a_res in self.antigen_residues:
                if self._min_distance(h_res, a_res) <= self.interface_cutoff:
                    self.heavy_interface.add((h_res.id[1], h_res.id[2]))
                    self.antigen_interface.add((a_res.id[1], a_res.id[2]))
                    
        # Calculate interface between light chain and antigen
        for l_res in self.light_residues:
            for a_res in self.antigen_residues:
                if self._min_distance(l_res, a_res) <= self.interface_cutoff:
                    self.light_interface.add((l_res.id[1], l_res.id[2]))
                    self.antigen_interface.add((a_res.id[1], a_res.id[2]))
        
        # Convert to sorted lists
        self.heavy_interface = sorted(list(self.heavy_interface))
        self.light_interface = sorted(list(self.light_interface))
        self.antigen_interface = sorted(list(self.antigen_interface))
        
        print(f"Heavy chain interface: {len(self.heavy_interface)} residues")
        print(f"Light chain interface: {len(self.light_interface)} residues")
        print(f"Antigen interface: {len(self.antigen_interface)} residues")
        
        # Save interface residues to file
        with open(f"{self.output_dir}/data/interface_residues.csv", 'w') as f:
            f.write("chain,residue_number,insertion_code\n")
            for res_id in self.heavy_interface:
                f.write(f"H,{res_id[0]},{res_id[1] if res_id[1] != ' ' else ''}\n")
            for res_id in self.light_interface:
                f.write(f"L,{res_id[0]},{res_id[1] if res_id[1] != ' ' else ''}\n")
            for res_id in self.antigen_interface:
                f.write(f"A,{res_id[0]},{res_id[1] if res_id[1] != ' ' else ''}\n")
        
        return self.heavy_interface, self.light_interface, self.antigen_interface

    def analyze_cdr_distribution(self):
        """Analyze distribution of interface residues in CDRs vs framework"""
        print("Analyzing CDR distribution...")
        
        # Classify heavy chain interface by CDR
        self.heavy_interface_by_region = {}
        for cdr_name, cdr_range in self.heavy_cdrs.items():
            self.heavy_interface_by_region[cdr_name] = [
                res for res in self.heavy_interface 
                if res[0] in cdr_range
            ]
        
        # Non-CDR residues
        all_heavy_cdrs = set()
        for cdr_range in self.heavy_cdrs.values():
            all_heavy_cdrs.update(cdr_range)
            
        self.heavy_interface_by_region['Framework'] = [
            res for res in self.heavy_interface 
            if res[0] not in all_heavy_cdrs
        ]
        
        # Classify light chain interface by CDR
        self.light_interface_by_region = {}
        for cdr_name, cdr_range in self.light_cdrs.items():
            self.light_interface_by_region[cdr_name] = [
                res for res in self.light_interface 
                if res[0] in cdr_range
            ]
        
        # Non-CDR residues
        all_light_cdrs = set()
        for cdr_range in self.light_cdrs.values():
            all_light_cdrs.update(cdr_range)
            
        self.light_interface_by_region['Framework'] = [
            res for res in self.light_interface 
            if res[0] not in all_light_cdrs
        ]
        
        # Calculate distribution of interface residues
        heavy_cdr_count = sum(len(residues) for region, residues 
                              in self.heavy_interface_by_region.items() 
                              if region != 'Framework')
        heavy_framework_count = len(self.heavy_interface_by_region['Framework'])
        heavy_total = heavy_cdr_count + heavy_framework_count
        
        light_cdr_count = sum(len(residues) for region, residues 
                             in self.light_interface_by_region.items() 
                             if region != 'Framework')
        light_framework_count = len(self.light_interface_by_region['Framework'])
        light_total = light_cdr_count + light_framework_count
        
        # Save CDR distribution to file
        with open(f"{self.output_dir}/data/cdr_distribution.csv", 'w') as f:
            f.write("chain,region,residue_count,percentage\n")
            
            if heavy_total > 0:
                for region, residues in self.heavy_interface_by_region.items():
                    percentage = len(residues) / heavy_total * 100 if heavy_total > 0 else 0
                    f.write(f"H,{region},{len(residues)},{percentage:.1f}\n")
            
            if light_total > 0:
                for region, residues in self.light_interface_by_region.items():
                    percentage = len(residues) / light_total * 100 if light_total > 0 else 0
                    f.write(f"L,{region},{len(residues)},{percentage:.1f}\n")
                    
        # Create a report summary
        with open(f"{self.output_dir}/data/cdr_summary.txt", 'w') as f:
            f.write("# CDR Distribution of Interface Residues\n\n")
            
            if heavy_total > 0:
                f.write(f"Heavy chain CDR residues: {heavy_cdr_count} ({heavy_cdr_count/heavy_total:.1%})\n")
                f.write(f"Heavy chain framework residues: {heavy_framework_count} ({heavy_framework_count/heavy_total:.1%})\n\n")
                
                f.write("Heavy chain breakdown by region:\n")
                for region, residues in self.heavy_interface_by_region.items():
                    if residues:
                        percentage = len(residues) / heavy_total * 100
                        f.write(f"- {region}: {len(residues)} residues ({percentage:.1f}%)\n")
            
            f.write("\n")
            if light_total > 0:
                f.write(f"Light chain CDR residues: {light_cdr_count} ({light_cdr_count/light_total:.1%})\n")
                f.write(f"Light chain framework residues: {light_framework_count} ({light_framework_count/light_total:.1%})\n\n")
                
                f.write("Light chain breakdown by region:\n")
                for region, residues in self.light_interface_by_region.items():
                    if residues:
                        percentage = len(residues) / light_total * 100
                        f.write(f"- {region}: {len(residues)} residues ({percentage:.1f}%)\n")
        
        print("CDR distribution analysis complete")
        return self.heavy_interface_by_region, self.light_interface_by_region

    def calculate_contacts(self):
        """Calculate detailed contact maps and interaction strengths"""
        print(f"Computing detailed contact maps (cutoff: {self.contact_cutoff}Å)...")
        
        # Initialize distance matrices with infinity
        heavy_antigen_distances = np.ones((len(self.heavy_residues), len(self.antigen_residues))) * np.inf
        light_antigen_distances = np.ones((len(self.light_residues), len(self.antigen_residues))) * np.inf
        
        # Calculate minimum distances between residue pairs
        for i, h_res in enumerate(self.heavy_residues):
            for j, a_res in enumerate(self.antigen_residues):
                distance = self._min_distance(h_res, a_res)
                heavy_antigen_distances[i, j] = distance
        
        for i, l_res in enumerate(self.light_residues):
            for j, a_res in enumerate(self.antigen_residues):
                distance = self._min_distance(l_res, a_res)
                light_antigen_distances[i, j] = distance
        
        # Create contact strength matrices (stronger for closer contacts)
        # Using sigmoid transformation: 1/(1+exp(distance-midpoint))
        heavy_antigen_contacts = np.zeros_like(heavy_antigen_distances)
        light_antigen_contacts = np.zeros_like(light_antigen_distances)
        
        mask_h = heavy_antigen_distances <= self.contact_cutoff
        mask_l = light_antigen_distances <= self.contact_cutoff
        
        heavy_antigen_contacts[mask_h] = 1.0 / (1.0 + np.exp(heavy_antigen_distances[mask_h] - 4.0))
        light_antigen_contacts[mask_l] = 1.0 / (1.0 + np.exp(light_antigen_distances[mask_l] - 4.0))
        
        # Calculate binding scores (sum of contact strengths)
        heavy_binding_scores = np.sum(heavy_antigen_contacts, axis=1)
        light_binding_scores = np.sum(light_antigen_contacts, axis=1)
        antigen_binding_scores = np.sum(np.vstack([
            np.sum(heavy_antigen_contacts, axis=0),
            np.sum(light_antigen_contacts, axis=0)
        ]), axis=0)
        
        # Store as object attributes
        self.heavy_antigen_distances = heavy_antigen_distances
        self.light_antigen_distances = light_antigen_distances
        self.heavy_antigen_contacts = heavy_antigen_contacts
        self.light_antigen_contacts = light_antigen_contacts
        self.heavy_binding_scores = heavy_binding_scores
        self.light_binding_scores = light_binding_scores
        self.antigen_binding_scores = antigen_binding_scores
        
        # Create binary contact maps (1 for contact, 0 for no contact)
        self.heavy_antigen_binary = (heavy_antigen_distances <= self.interface_cutoff).astype(int)
        self.light_antigen_binary = (light_antigen_distances <= self.interface_cutoff).astype(int)
        
        # Save contact data
        np.save(f"{self.output_dir}/data/heavy_antigen_contacts.npy", heavy_antigen_contacts)
        np.save(f"{self.output_dir}/data/light_antigen_contacts.npy", light_antigen_contacts)
        
        # Save binding scores
        np.save(f"{self.output_dir}/data/heavy_binding_scores.npy", heavy_binding_scores)
        np.save(f"{self.output_dir}/data/light_binding_scores.npy", light_binding_scores)
        np.save(f"{self.output_dir}/data/antigen_binding_scores.npy", antigen_binding_scores)
        
        # Save as CSV for easy importing
        pd.DataFrame({
            'residue': range(1, len(heavy_binding_scores) + 1),
            'score': heavy_binding_scores
        }).to_csv(f"{self.output_dir}/data/heavy_binding_scores.csv", index=False)
        
        pd.DataFrame({
            'residue': range(1, len(light_binding_scores) + 1),
            'score': light_binding_scores
        }).to_csv(f"{self.output_dir}/data/light_binding_scores.csv", index=False)
        
        pd.DataFrame({
            'residue': range(1, len(antigen_binding_scores) + 1),
            'score': antigen_binding_scores
        }).to_csv(f"{self.output_dir}/data/antigen_binding_scores.csv", index=False)
        
        # Identify key interacting residues (top contributors)
        self.key_residues = {}
        
        h_indices = np.argsort(heavy_binding_scores)[::-1][:10]
        self.key_residues['heavy'] = [(self.heavy_residues[i].id[1], 
                                      three_to_one(self.heavy_residues[i].get_resname()),
                                      heavy_binding_scores[i]) for i in h_indices if heavy_binding_scores[i] > 0]
        
        l_indices = np.argsort(light_binding_scores)[::-1][:10]
        self.key_residues['light'] = [(self.light_residues[i].id[1],
                                      three_to_one(self.light_residues[i].get_resname()),
                                      light_binding_scores[i]) for i in l_indices if light_binding_scores[i] > 0]
        
        ag_indices = np.argsort(antigen_binding_scores)[::-1][:10]
        self.key_residues['antigen'] = [(self.antigen_residues[i].id[1],
                                        three_to_one(self.antigen_residues[i].get_resname()),
                                        antigen_binding_scores[i]) for i in ag_indices if antigen_binding_scores[i] > 0]
        
        # Save key residues
        with open(f"{self.output_dir}/data/key_residues.txt", 'w') as f:
            f.write("# Key Binding Residues\n\n")
            
            f.write("## Heavy Chain\n")
            for res in self.key_residues['heavy']:
                f.write(f"Residue {res[0]} ({res[1]}): Score {res[2]:.3f}\n")
            f.write("\n")
            
            f.write("## Light Chain\n")
            for res in self.key_residues['light']:
                f.write(f"Residue {res[0]} ({res[1]}): Score {res[2]:.3f}\n")
            f.write("\n")
            
            f.write("## Antigen\n")
            for res in self.key_residues['antigen']:
                f.write(f"Residue {res[0]} ({res[1]}): Score {res[2]:.3f}\n")
                
        print(f"Contact analysis complete - found {np.sum(self.heavy_antigen_binary)} heavy chain contacts and " + 
              f"{np.sum(self.light_antigen_binary)} light chain contacts")
        return heavy_binding_scores, light_binding_scores, antigen_binding_scores

    def calculate_binding_energy(self):
        """Calculate simplified binding energy based on contact distances"""
        print("Calculating binding energy estimates...")
        
        # Simple distance-based energy model
        # E = -1.0 / (1.0 + exp(distance - midpoint))
        
        contact_energies = {}
        total_energy = 0
        
        # Calculate heavy chain contribution
        heavy_energy = 0
        for i, h_res in enumerate(self.heavy_residues):
            h_res_id = (h_res.id[1], h_res.id[2])
            if h_res_id in self.heavy_interface:
                for j, a_res in enumerate(self.antigen_residues):
                    a_res_id = (a_res.id[1], a_res.id[2])
                    if a_res_id in self.antigen_interface:
                        dist = self.heavy_antigen_distances[i, j]
                        if dist <= self.contact_cutoff:
                            # Simplified energy function (arbitrary units)
                            if dist < 2.0:  # Too close, repulsive
                                energy = 10.0  # Repulsive
                            else:
                                energy = -1.0 * (1.0 / (1.0 + np.exp(dist - 4.0)))
                            
                            contact_energies[(f"H:{h_res.id[1]}", f"A:{a_res.id[1]}")] = energy
                            heavy_energy += energy
                            total_energy += energy
        
        # Calculate light chain contribution
        light_energy = 0
        for i, l_res in enumerate(self.light_residues):
            l_res_id = (l_res.id[1], l_res.id[2])
            if l_res_id in self.light_interface:
                for j, a_res in enumerate(self.antigen_residues):
                    a_res_id = (a_res.id[1], a_res.id[2])
                    if a_res_id in self.antigen_interface:
                        dist = self.light_antigen_distances[i, j]
                        if dist <= self.contact_cutoff:
                            # Simplified energy function
                            if dist < 2.0:  # Too close, repulsive
                                energy = 10.0  # Repulsive
                            else:
                                energy = -1.0 * (1.0 / (1.0 + np.exp(dist - 4.0)))
                            
                            contact_energies[(f"L:{l_res.id[1]}", f"A:{a_res.id[1]}")] = energy
                            light_energy += energy
                            total_energy += energy
        
        # Sort pairwise energies by strength
        sorted_energies = sorted(contact_energies.items(), key=lambda x: x[1])
        top_contacts = sorted_energies[:10]  # Top 10 strongest interactions
        
        # Store results
        self.binding_energy = {
            'total': total_energy,
            'heavy_chain': heavy_energy,
            'light_chain': light_energy,
            'top_contacts': top_contacts,
            'all_contacts': dict(sorted_energies)
        }
        
        # Save binding energy data
        with open(f"{self.output_dir}/data/binding_energy.txt", 'w') as f:
            f.write("# Binding Energy Analysis\n\n")
            
            f.write(f"Total binding energy: {total_energy:.3f}\n")
            f.write(f"Heavy chain contribution: {heavy_energy:.3f}\n")
            f.write(f"Light chain contribution: {light_energy:.3f}\n\n")
            
            f.write("## Top Energetic Contacts\n")
            for contact, value in top_contacts:
                f.write(f"{contact[0]} - {contact[1]}: {value:.3f}\n")
        
        print(f"Binding energy calculation complete: {total_energy:.2f} total energy units")
        return total_energy, heavy_energy, light_energy

    def analyze_interactions(self):
        """Analyze specific interactions (H-bonds, salt bridges, etc.)"""
        print("Analyzing specific interactions...")
        
        # Hydrogen bonds
        h_bonds = self._find_hydrogen_bonds()
        
        # Salt bridges
        salt_bridges = self._find_salt_bridges()
        
        # π-π stacking and cation-π interactions
        pi_stacking, cation_pi = self._find_aromatic_interactions()
        
        # Save interactions data
        with open(f"{self.output_dir}/data/specific_interactions.txt", 'w', encoding='utf-8') as f:
            f.write("# Specific Interactions in the Interface\n\n")
            
            f.write(f"## Hydrogen Bonds ({len(h_bonds)})\n")
            for donor, acceptor, dist in h_bonds:
                f.write(f"{donor} -> {acceptor}: {dist:.2f} Å\n")
        
            f.write(f"\n## Salt Bridges ({len(salt_bridges)})\n")
            for res1, res2, dist in salt_bridges:
                f.write(f"{res1} <-> {res2}: {dist:.2f} Å\n")
        
            f.write(f"\n## π-π Stacking ({len(pi_stacking)})\n")
            for res1, res2, dist in pi_stacking:
                f.write(f"{res1} <-> {res2}: {dist:.2f} Å\n")
        
            f.write(f"\n## Cation-π Interactions ({len(cation_pi)})\n")
            for res1, res2, dist in cation_pi:
                f.write(f"{res1} <-> {res2}: {dist:.2f} Å\n")
    
        # Create CSV for each interaction type
        self._save_interactions_csv(h_bonds, "hydrogen_bonds")
        self._save_interactions_csv(salt_bridges, "salt_bridges")
        self._save_interactions_csv(pi_stacking, "pi_stacking")
        self._save_interactions_csv(cation_pi, "cation_pi")
        
        self.interactions = {
            'hydrogen_bonds': h_bonds,
            'salt_bridges': salt_bridges,
            'pi_stacking': pi_stacking,
            'cation_pi': cation_pi
        }
        
        print(f"Found {len(h_bonds)} hydrogen bonds, {len(salt_bridges)} salt bridges, " +
              f"{len(pi_stacking)} π-π stacking and {len(cation_pi)} cation-π interactions")
        return self.interactions
    
    def _save_interactions_csv(self, interactions, name):
        """Save interactions to CSV file"""
        with open(f"{self.output_dir}/data/{name}.csv", 'w', encoding='utf-8') as f:
            f.write("residue1,residue2,distance\n")
            for res1, res2, dist in interactions:
                f.write(f"{res1},{res2},{dist:.2f}\n")
                
    def _find_hydrogen_bonds(self):
        """Identify hydrogen bonds in the interface"""
        h_bonds = []
        
        # Helper function to identify potential donors/acceptors
        def is_donor(atom):
            return atom.element in ['N', 'O'] and any(a.element == 'H' for a in atom.get_parent())
        
        def is_acceptor(atom):
            return atom.element in ['O', 'N']
        
        # Check heavy chain - antigen
        for h_res in self.heavy_residues:
            h_res_id = (h_res.id[1], h_res.id[2])
            if h_res_id in self.heavy_interface:
                for a_res in self.antigen_residues:
                    a_res_id = (a_res.id[1], a_res.id[2])
                    if a_res_id in self.antigen_interface:
                        # Check heavy chain donors to antigen acceptors
                        for h_atom in h_res:
                            if is_donor(h_atom):
                                for a_atom in a_res:
                                    if is_acceptor(a_atom):
                                        dist = h_atom - a_atom
                                        if dist <= 3.5:
                                            h_bonds.append((
                                                f"H:{h_res.id[1]}.{h_atom.name}",
                                                f"A:{a_res.id[1]}.{a_atom.name}",
                                                dist
                                            ))
                        
                        # Check antigen donors to heavy chain acceptors
                        for a_atom in a_res:
                            if is_donor(a_atom):
                                for h_atom in h_res:
                                    if is_acceptor(h_atom):
                                        dist = a_atom - h_atom
                                        if dist <= 3.5:
                                            h_bonds.append((
                                                f"A:{a_res.id[1]}.{a_atom.name}",
                                                f"H:{h_res.id[1]}.{h_atom.name}",
                                                dist
                                            ))
        
        # Check light chain - antigen (same logic)
        for l_res in self.light_residues:
            l_res_id = (l_res.id[1], l_res.id[2])
            if l_res_id in self.light_interface:
                for a_res in self.antigen_residues:
                    a_res_id = (a_res.id[1], a_res.id[2])
                    if a_res_id in self.antigen_interface:
                        for l_atom in l_res:
                            if is_donor(l_atom):
                                for a_atom in a_res:
                                    if is_acceptor(a_atom):
                                        dist = l_atom - a_atom
                                        if dist <= 3.5:
                                            h_bonds.append((
                                                f"L:{l_res.id[1]}.{l_atom.name}",
                                                f"A:{a_res.id[1]}.{a_atom.name}",
                                                dist
                                            ))
                        
                        for a_atom in a_res:
                            if is_donor(a_atom):
                                for l_atom in l_res:
                                    if is_acceptor(l_atom):
                                        dist = a_atom - l_atom
                                        if dist <= 3.5:
                                            h_bonds.append((
                                                f"A:{a_res.id[1]}.{a_atom.name}",
                                                f"L:{l_res.id[1]}.{l_atom.name}",
                                                dist
                                            ))
        
        # Sort by distance
        h_bonds.sort(key=lambda x: x[2])
        return h_bonds
    
    def _find_salt_bridges(self):
        """Identify salt bridges in the interface"""
        salt_bridges = []
        
        # Identify charged residues
        pos_charged = ['ARG', 'LYS', 'HIS']
        neg_charged = ['ASP', 'GLU']
        
        # Check heavy chain - antigen
        for h_res in self.heavy_residues:
            h_res_id = (h_res.id[1], h_res.id[2])
            if h_res_id in self.heavy_interface:
                h_charge = None
                if h_res.get_resname() in pos_charged:
                    h_charge = 'positive'
                elif h_res.get_resname() in neg_charged:
                    h_charge = 'negative'
                
                if h_charge:
                    for a_res in self.antigen_residues:
                        a_res_id = (a_res.id[1], a_res.id[2])
                        if a_res_id in self.antigen_interface:
                            a_charge = None
                            if a_res.get_resname() in pos_charged:
                                a_charge = 'positive'
                            elif a_res.get_resname() in neg_charged:
                                a_charge = 'negative'
                            
                            if a_charge and h_charge != a_charge:
                                dist = self._min_distance(h_res, a_res)
                                if dist <= 4.0:  # Standard salt bridge distance
                                    salt_bridges.append((
                                        f"H:{h_res.id[1]}({h_res.get_resname()})",
                                        f"A:{a_res.id[1]}({a_res.get_resname()})",
                                        dist
                                    ))
        
        # Check light chain - antigen
        for l_res in self.light_residues:
            l_res_id = (l_res.id[1], l_res.id[2])
            if l_res_id in self.light_interface:
                l_charge = None
                if l_res.get_resname() in pos_charged:
                    l_charge = 'positive'
                elif l_res.get_resname() in neg_charged:
                    l_charge = 'negative'
                
                if l_charge:
                    for a_res in self.antigen_residues:
                        a_res_id = (a_res.id[1], a_res.id[2])
                        if a_res_id in self.antigen_interface:
                            a_charge = None
                            if a_res.get_resname() in pos_charged:
                                a_charge = 'positive'
                            elif a_res.get_resname() in neg_charged:
                                a_charge = 'negative'
                            
                            if a_charge and l_charge != a_charge:
                                dist = self._min_distance(l_res, a_res)
                                if dist <= 4.0:
                                    salt_bridges.append((
                                        f"L:{l_res.id[1]}({l_res.get_resname()})",
                                        f"A:{a_res.id[1]}({a_res.get_resname()})",
                                        dist
                                    ))
        
        # Sort by distance
        salt_bridges.sort(key=lambda x: x[2])
        return salt_bridges
    
    def _find_aromatic_interactions(self):
        """Identify π-π stacking and cation-π interactions"""
        pi_stacking = []
        cation_pi = []
        
        # Identify aromatic and cationic residues
        aromatic = ['PHE', 'TYR', 'TRP', 'HIS']
        cationic = ['ARG', 'LYS']
        
        # Check heavy chain - antigen
        for h_res in self.heavy_residues:
            h_res_id = (h_res.id[1], h_res.id[2])
            if h_res_id in self.heavy_interface:
                h_is_aromatic = h_res.get_resname() in aromatic
                h_is_cationic = h_res.get_resname() in cationic
                
                for a_res in self.antigen_residues:
                    a_res_id = (a_res.id[1], a_res.id[2])
                    if a_res_id in self.antigen_interface:
                        a_is_aromatic = a_res.get_resname() in aromatic
                        a_is_cationic = a_res.get_resname() in cationic
                        
                        dist = self._min_distance(h_res, a_res)
                        
                        # Check for π-π stacking
                        if h_is_aromatic and a_is_aromatic and dist <= 6.0:
                            pi_stacking.append((
                                f"H:{h_res.id[1]}({h_res.get_resname()})",
                                f"A:{a_res.id[1]}({a_res.get_resname()})",
                                dist
                            ))
                        
                        # Check for cation-π interactions
                        elif (h_is_aromatic and a_is_cationic) or (h_is_cationic and a_is_aromatic):
                            if dist <= 6.0:
                                cation_pi.append((
                                    f"H:{h_res.id[1]}({h_res.get_resname()})",
                                    f"A:{a_res.id[1]}({a_res.get_resname()})",
                                    dist
                                ))
        
        # Check light chain - antigen
        for l_res in self.light_residues:
            l_res_id = (l_res.id[1], l_res.id[2])
            if l_res_id in self.light_interface:
                l_is_aromatic = l_res.get_resname() in aromatic
                l_is_cationic = l_res.get_resname() in cationic
                
                for a_res in self.antigen_residues:
                    a_res_id = (a_res.id[1], a_res.id[2])
                    if a_res_id in self.antigen_interface:
                        a_is_aromatic = a_res.get_resname() in aromatic
                        a_is_cationic = a_res.get_resname() in cationic
                        
                        dist = self._min_distance(l_res, a_res)
                        
                        # Check for π-π stacking
                        if l_is_aromatic and a_is_aromatic and dist <= 6.0:
                            pi_stacking.append((
                                f"L:{l_res.id[1]}({l_res.get_resname()})",
                                f"A:{a_res.id[1]}({a_res.get_resname()})",
                                dist
                            ))
                        
                        # Check for cation-π interactions
                        elif (l_is_cationic and a_is_aromatic) or (l_is_aromatic and a_is_cationic):
                            if dist <= 6.0:
                                cation_pi.append((
                                    f"L:{l_res.id[1]}({l_res.get_resname()})",
                                    f"A:{a_res.id[1]}({a_res.get_resname()})",
                                    dist
                                ))
        
        # Sort by distance
        pi_stacking.sort(key=lambda x: x[2])
        cation_pi.sort(key=lambda x: x[2])
        
        return pi_stacking, cation_pi

    def analyze_interface_properties(self):
        """Analyze physicochemical properties of the interface"""
        print("Analyzing interface physicochemical properties...")
    
        # Amino acid properties
        aa_properties = {
            'A': 'hydrophobic', 'V': 'hydrophobic', 'L': 'hydrophobic',
            'I': 'hydrophobic', 'M': 'hydrophobic', 'F': 'hydrophobic',
            'W': 'hydrophobic', 'P': 'hydrophobic',
            'D': 'charged-', 'E': 'charged-',
            'K': 'charged+', 'R': 'charged+', 'H': 'charged+',
            'S': 'polar', 'T': 'polar', 'N': 'polar', 'Q': 'polar', 'Y': 'polar',
            'C': 'special', 'G': 'special'
        }
        
        # Initialize property counts
        props = {}
        for chain_name in ['heavy', 'light', 'antigen']:
            props[chain_name] = {
                'hydrophobic': 0, 'charged+': 0, 'charged-': 0, 
                'polar': 0, 'special': 0, 'total': 0
            }
        
        # Analyze heavy chain interface
        for res_id in self.heavy_interface:
            for res in self.heavy_residues:
                if (res.id[1], res.id[2]) == res_id:
                    try:
                        aa = three_to_one(res.get_resname())
                        prop = aa_properties.get(aa, 'unknown')
                        props['heavy'][prop] = props['heavy'].get(prop, 0) + 1
                        props['heavy']['total'] += 1
                    except:
                        pass  # Skip non-standard residues
                    break
        
        # Analyze light chain interface
        for res_id in self.light_interface:
            for res in self.light_residues:
                if (res.id[1], res.id[2]) == res_id:
                    try:
                        aa = three_to_one(res.get_resname())
                        prop = aa_properties.get(aa, 'unknown')
                        props['light'][prop] = props['light'].get(prop, 0) + 1
                        props['light']['total'] += 1
                    except:
                        pass
                    break
        
        # Analyze antigen interface
        for res_id in self.antigen_interface:
            for res in self.antigen_residues:
                if (res.id[1], res.id[2]) == res_id:
                    try:
                        aa = three_to_one(res.get_resname())
                        prop = aa_properties.get(aa, 'unknown')
                        props['antigen'][prop] = props['antigen'].get(prop, 0) + 1
                        props['antigen']['total'] += 1
                    except:
                        pass
                    break
        
        # Save the property analysis
        with open(f"{self.output_dir}/data/interface_properties.csv", 'w') as f:
            f.write("chain,property,count,percentage\n")
            for chain, properties in props.items():
                total = properties['total']
                for prop, count in properties.items():
                    if prop != 'total' and count > 0:
                        percentage = count / total * 100 if total > 0 else 0
                        f.write(f"{chain},{prop},{count},{percentage:.1f}\n")
        
        self.interface_properties = props
        
        # Calculate shape complementarity (simplified)
        # Calculate distances between interface residues
        interface_distances = []
        
        for h_res in self.heavy_residues:
            h_res_id = (h_res.id[1], h_res.id[2])
            if h_res_id in self.heavy_interface:
                for a_res in self.antigen_residues:
                    a_res_id = (a_res.id[1], a_res.id[2])
                    if a_res_id in self.antigen_interface:
                        dist = self._min_distance(h_res, a_res)
                        interface_distances.append(dist)
        
        for l_res in self.light_residues:
            l_res_id = (l_res.id[1], l_res.id[2])
            if l_res_id in self.light_interface:
                for a_res in self.antigen_residues:
                    a_res_id = (a_res.id[1], a_res.id[2])
                    if a_res_id in self.antigen_interface:
                        dist = self._min_distance(l_res, a_res)
                        interface_distances.append(dist)
        
        # Calculate complementarity metrics
        if interface_distances:
            avg_distance = np.mean(interface_distances)
            min_distance = np.min(interface_distances)
            max_distance = np.max(interface_distances)
            std_distance = np.std(interface_distances)
            
            # Simple complementarity score: 1/(1+avg_distance)
            # Higher score means better complementarity
            sc_score = 1.0 / (1.0 + avg_distance)
        else:
            avg_distance = min_distance = max_distance = std_distance = sc_score = 0
        
        self.shape_complementarity = {
            'score': sc_score,
            'avg_distance': avg_distance,
            'min_distance': min_distance,
            'max_distance': max_distance,
            'std_distance': std_distance
        }
        
        with open(f"{self.output_dir}/data/shape_complementarity.txt", 'w') as f:
            f.write("# Interface Shape Complementarity\n\n")
            f.write(f"Complementarity score: {sc_score:.3f}\n")
            f.write(f"Average interface distance: {avg_distance:.2f} Å\n")
            f.write(f"Minimum interface distance: {min_distance:.2f} Å\n")
            f.write(f"Maximum interface distance: {max_distance:.2f} Å\n")
            f.write(f"Distance standard deviation: {std_distance:.2f} Å\n")
        
        print(f"Interface property analysis complete (Shape complementarity: {sc_score:.3f})")
        return self.interface_properties, self.shape_complementarity

    def visualize_results(self):
        """Create plots of the analysis results"""
        print("Creating visualizations...")
        
        # Plot 1: Contact maps
        plt.figure(figsize=(12, 10))
        plt.subplot(2, 1, 1)
        sns.heatmap(self.heavy_antigen_contacts, cmap='viridis')
        plt.title('Heavy Chain-Antigen Contact Strength Map')
        plt.xlabel('Antigen Residue Index')
        plt.ylabel('Heavy Chain Residue Index')
        
        plt.subplot(2, 1, 2)
        sns.heatmap(self.light_antigen_contacts, cmap='viridis')
        plt.title('Light Chain-Antigen Contact Strength Map')
        plt.xlabel('Antigen Residue Index')
        plt.ylabel('Light Chain Residue Index')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/contact_maps.png", dpi=300)
        
        # Plot 2: Binding scores with CDR regions highlighted
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 1, 1)
        plt.bar(range(1, len(self.heavy_binding_scores) + 1), self.heavy_binding_scores)
        plt.title('Heavy Chain-Antigen Binding Strength by Residue')
        plt.xlabel('Heavy Chain Residue Index')
        plt.ylabel('Binding Strength')
        
        # Highlight CDRs
        colors = ['lightgreen', 'lightblue', 'salmon']
        for i, (cdr_name, cdr_range) in enumerate(self.heavy_cdrs.items()):
            plt.axvspan(min(cdr_range), max(cdr_range), alpha=0.3, color=colors[i], label=cdr_name)
        
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.bar(range(1, len(self.light_binding_scores) + 1), self.light_binding_scores)
        plt.title('Light Chain-Antigen Binding Strength by Residue')
        plt.xlabel('Light Chain Residue Index')
        plt.ylabel('Binding Strength')
        
        # Highlight CDRs
        for i, (cdr_name, cdr_range) in enumerate(self.light_cdrs.items()):
            plt.axvspan(min(cdr_range), max(cdr_range), alpha=0.3, color=colors[i], label=cdr_name)
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/binding_scores.png", dpi=300)
        
        # Plot 3: Antigen binding site
        plt.figure(figsize=(10, 5))
        plt.bar(range(1, len(self.antigen_binding_scores) + 1), self.antigen_binding_scores)
        plt.title('Antigen-Antibody Binding Strength by Residue')
        plt.xlabel('Antigen Residue Index')
        plt.ylabel('Binding Strength')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/antigen_binding.png", dpi=300)
        
        # Plot 4: Interface properties pie charts
        props = self.interface_properties
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Heavy chain
        heavy_data = [(k, v) for k, v in props['heavy'].items() if k != 'total' and v > 0]
        if heavy_data:
            labels, values = zip(*heavy_data)
            axes[0].pie(values, labels=labels, autopct='%1.1f%%')
            axes[0].set_title('Heavy Chain Interface Composition')
        
        # Light chain
        light_data = [(k, v) for k, v in props['light'].items() if k != 'total' and v > 0]
        if light_data:
            labels, values = zip(*light_data)
            axes[1].pie(values, labels=labels, autopct='%1.1f%%')
            axes[1].set_title('Light Chain Interface Composition')
        
        # Antigen
        antigen_data = [(k, v) for k, v in props['antigen'].items() if k != 'total' and v > 0]
        if antigen_data:
            labels, values = zip(*antigen_data)
            axes[2].pie(values, labels=labels, autopct='%1.1f%%')
            axes[2].set_title('Antigen Interface Composition')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/interface_composition.png", dpi=300)
        
        # Plot 5: CDR distribution
        plt.figure(figsize=(12, 5))
        
        # Heavy chain
        heavy_data = {k: len(v) for k, v in self.heavy_interface_by_region.items() if len(v) > 0}
        if heavy_data:
            plt.subplot(1, 2, 1)
            plt.bar(heavy_data.keys(), heavy_data.values())
            plt.title('Heavy Chain Interface Distribution')
            plt.xlabel('Region')
            plt.ylabel('Number of Residues')
            plt.xticks(rotation=45)
        
        # Light chain
        light_data = {k: len(v) for k, v in self.light_interface_by_region.items() if len(v) > 0}
        if light_data:
            plt.subplot(1, 2, 2)
            plt.bar(light_data.keys(), light_data.values())
            plt.title('Light Chain Interface Distribution')
            plt.xlabel('Region')
            plt.ylabel('Number of Residues')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/cdr_distribution.png", dpi=300)
        
        print("Visualization complete")

    def create_summary_report(self):
        """Generate a comprehensive summary report"""
        print("Creating summary report...")
        
        with open(f"{self.output_dir}/summary_report.txt", 'w', encoding='utf-8') as f:
            f.write("# ANTIBODY-ANTIGEN INTERFACE ANALYSIS SUMMARY\n\n")
            
            f.write("## 1. OVERVIEW\n")
            f.write(f"PDB File: {self.pdb_file}\n")
            f.write(f"Heavy Chain: {self.heavy_id} ({len(self.heavy_seq)} residues)\n")
            f.write(f"Light Chain: {self.light_id} ({len(self.light_seq)} residues)\n")
            f.write(f"Antigen: {self.antigen_id} ({len(self.antigen_seq)} residues)\n\n")
            
            f.write("## 2. INTERFACE STATISTICS\n")
            f.write(f"Heavy chain interface residues: {len(self.heavy_interface)}\n")
            f.write(f"Light chain interface residues: {len(self.light_interface)}\n")
            f.write(f"Antigen interface residues: {len(self.antigen_interface)}\n\n")
            
            # CDR distribution
            heavy_cdr_count = sum(len(residues) for region, residues 
                                 in self.heavy_interface_by_region.items() 
                                 if region != 'Framework')
            heavy_framework_count = len(self.heavy_interface_by_region['Framework'])
            heavy_total = heavy_cdr_count + heavy_framework_count
            
            light_cdr_count = sum(len(residues) for region, residues 
                                 in self.light_interface_by_region.items() 
                                 if region != 'Framework')
            light_framework_count = len(self.light_interface_by_region['Framework'])
            light_total = light_cdr_count + light_framework_count
            
            f.write("### Interface Distribution\n")
            if heavy_total > 0:
                f.write(f"Heavy chain CDR residues: {heavy_cdr_count} ({heavy_cdr_count/heavy_total:.1%})\n")
                f.write(f"Heavy chain framework residues: {heavy_framework_count} ({heavy_framework_count/heavy_total:.1%})\n")
            
            if light_total > 0:
                f.write(f"Light chain CDR residues: {light_cdr_count} ({light_cdr_count/light_total:.1%})\n")
                f.write(f"Light chain framework residues: {light_framework_count} ({light_framework_count/light_total:.1%})\n\n")
            
            f.write("### Interface by Region\n")
            for region, residues in self.heavy_interface_by_region.items():
                if residues:
                    f.write(f"Heavy {region}: {len(residues)} residues - {', '.join(str(r[0]) for r in residues)}\n")
            
            f.write("\n")
            for region, residues in self.light_interface_by_region.items():
                if residues:
                    f.write(f"Light {region}: {len(residues)} residues - {', '.join(str(r[0]) for r in residues)}\n")
            
            f.write("\n## 3. BINDING ENERGY\n")
            energy = self.binding_energy
            f.write(f"Total binding energy: {energy['total']:.3f} (arbitrary units)\n")
            f.write(f"Heavy chain contribution: {energy['heavy_chain']:.3f} ({energy['heavy_chain']/energy['total']:.1%})\n")
            f.write(f"Light chain contribution: {energy['light_chain']:.3f} ({energy['light_chain']/energy['total']:.1%})\n\n")
            
            f.write("### Top Energetic Contacts\n")
            for i, (contact, value) in enumerate(energy['top_contacts']):
                f.write(f"{i+1}. {contact[0]} - {contact[1]}: {value:.3f}\n")
            
            f.write("\n## 4. INTERFACE PROPERTIES\n")
            props = self.interface_properties
            
            f.write("### Amino Acid Composition\n")
            f.write("Heavy chain interface:\n")
            for prop, count in props['heavy'].items():
                if prop != 'total' and count > 0:
                    f.write(f"  {prop}: {count} ({count/props['heavy']['total']:.1%})\n")
            
            f.write("\nLight chain interface:\n")
            for prop, count in props['light'].items():
                if prop != 'total' and count > 0:
                    f.write(f"  {prop}: {count} ({count/props['light']['total']:.1%})\n")
            
            f.write("\nAntigen interface:\n")
            for prop, count in props['antigen'].items():
                if prop != 'total' and count > 0:
                    f.write(f"  {prop}: {count} ({count/props['antigen']['total']:.1%})\n")
            
            f.write("\n### Shape Complementarity\n")
            sc = self.shape_complementarity
            f.write(f"Complementarity score: {sc['score']:.3f}\n")
            f.write(f"Average interface distance: {sc['avg_distance']:.2f} Å\n")
            f.write(f"Minimum interface distance: {sc['min_distance']:.2f} Å\n")
            f.write(f"Maximum interface distance: {sc['max_distance']:.2f} Å\n\n")
            
            f.write("## 5. SPECIFIC INTERACTIONS\n")
            f.write(f"### Hydrogen Bonds ({len(self.interactions['hydrogen_bonds'])})\n")
            for donor, acceptor, dist in self.interactions['hydrogen_bonds'][:10]:
                f.write(f"{donor} -> {acceptor}: {dist:.2f} Å\n")
                
            if len(self.interactions['hydrogen_bonds']) > 10:
                f.write(f"... and {len(self.interactions['hydrogen_bonds'])-10} more hydrogen bonds\n")
            
            f.write(f"\n### Salt Bridges ({len(self.interactions['salt_bridges'])})\n")
            for res1, res2, dist in self.interactions['salt_bridges']:
                f.write(f"{res1} <-> {res2}: {dist:.2f} Å\n")
        
            f.write(f"\n### π-π Stacking ({len(self.interactions['pi_stacking'])})\n")
            for res1, res2, dist in self.interactions['pi_stacking']:
                f.write(f"{res1} <-> {res2}: {dist:.2f} Å\n")
        
            f.write(f"\n### Cation-π Interactions ({len(self.interactions['cation_pi'])})\n")
            for res1, res2, dist in self.interactions['cation_pi']:
                f.write(f"{res1} <-> {res2}: {dist:.2f} Å\n")
        
            f.write("\n## 6. KEY BINDING RESIDUES\n")
            f.write("### Heavy Chain\n")
            for res in self.key_residues['heavy']:
                f.write(f"Residue {res[0]} ({res[1]}): Score {res[2]:.3f}\n")
            
            f.write("\n### Light Chain\n")
            for res in self.key_residues['light']:
                f.write(f"Residue {res[0]} ({res[1]}): Score {res[2]:.3f}\n")
            
            f.write("\n### Antigen\n")
            for res in self.key_residues['antigen']:
                f.write(f"Residue {res[0]} ({res[1]}): Score {res[2]:.3f}\n")
        
        print(f"Summary report saved to {self.output_dir}/summary_report.txt")

    def run_all(self):
        """Run the complete analysis pipeline"""
        self.extract_sequences()
        self.identify_interface()
        self.analyze_cdr_distribution()
        self.calculate_contacts()
        self.calculate_binding_energy()
        self.analyze_interactions()
        self.analyze_interface_properties()
        self.visualize_results()
        self.create_summary_report()
        print("\nAnalysis complete!")


def main():
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Analyze antibody-antigen interface from PDB structure")
    parser.add_argument("pdb_file", help="Path to PDB file")
    parser.add_argument("--heavy", default="H", help="Heavy chain ID (default: H)")
    parser.add_argument("--light", default="L", help="Light chain ID (default: L)")
    parser.add_argument("--antigen", default="C", help="Antigen chain ID (default: C)")
    parser.add_argument("--output", default="results/interface_analysis", help="Output directory")
    
    args = parser.parse_args()
    
    # Check if PDB file exists
    if not os.path.exists(args.pdb_file):
        print(f"Error: PDB file not found at {args.pdb_file}")
        sys.exit(1)
    
    try:
        analyzer = AntibodyInterface(
            args.pdb_file, 
            args.heavy, 
            args.light, 
            args.antigen, 
            args.output
        )
        analyzer.run_all()
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()