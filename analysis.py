#!/usr/bin/env python3

import os
import json
import shutil
import numpy as np
import pandas as pd
import argparse

# -------------------------
# RMSD Imports
# -------------------------
import MDAnalysis as mda
from MDAnalysis.analysis.rms import rmsd
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree
import math

# -------------------------
# For PDF generation
# -------------------------
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Table,
                                TableStyle, Spacer, Image as RLImage)
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# -----------------------------------------------------------------------
# Pipeline Code (pLDDT, RMSD, pTM/ipTM, PAE, Clash)
# -----------------------------------------------------------------------
def extract_plddt_scores(ranking_debug_json_path):
    results = []
    if not os.path.isfile(ranking_debug_json_path):
        return results

    object_name = os.path.basename(os.path.dirname(ranking_debug_json_path))
    folder_two_up = os.path.basename(os.path.dirname(os.path.dirname(ranking_debug_json_path)))
    name_col = folder_two_up + "/" + object_name

    try:
        with open(ranking_debug_json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {ranking_debug_json_path}: {e}")
        return results

    if "iptm+ptm" not in data:
        return results

    iptm_ptm = data["iptm+ptm"]
    items = list(iptm_ptm.items())
    for rank_idx, (model_key, plddt_val) in enumerate(items):
        ranked_file = f"ranked_{rank_idx}.pdb"
        results.append((name_col, object_name, model_key, ranked_file, plddt_val))
    return results


def calculate_rmsd(ref_pdb, af_pdb):
    try:
        ref_u = mda.Universe(ref_pdb)
        af_u  = mda.Universe(af_pdb)

        ref_atoms = ref_u.select_atoms('backbone')
        af_atoms  = af_u.select_atoms('backbone')

        if len(ref_atoms) == 0:
            print(f"Error: No backbone atoms found in reference PDB: {ref_pdb}")
            return "N/A"
        if len(af_atoms) == 0:
            print(f"Error: No backbone atoms found in AlphaFold PDB: {af_pdb}")
            return "N/A"
        if len(ref_atoms) != len(af_atoms):
            print(f"Warning: Atom count mismatch between {ref_pdb} and {af_pdb}.")
            return "N/A"

        rmsd_value = rmsd(
            af_atoms.positions,
            ref_atoms.positions,
            center=True,
            superposition=True
        )
        return rmsd_value
    except Exception as e:
        print(f"Error calculating RMSD between {ref_pdb} and {af_pdb}: {e}")
        return "N/A"


def extract_ptm_iptm(ranking_debug_filepath, model_identifier):
    pTM = np.nan
    ipTM = np.nan
    try:
        with open(ranking_debug_filepath, 'r') as f_rank:
            rank_data = json.load(f_rank)
    except Exception as e:
        print(f"Error reading {ranking_debug_filepath}: {e}")
        return pTM, ipTM

    if 'iptm+ptm' in rank_data:
        iptm_ptm_dict = rank_data['iptm+ptm']
        if model_identifier in iptm_ptm_dict:
            pTM = iptm_ptm_dict[model_identifier] * 100
            ipTM = pTM
        else:
            print(f"Warning: Model identifier '{model_identifier}' not found in 'iptm+ptm'.")
    else:
        print(f"Warning: 'iptm+ptm' key not found in {ranking_debug_filepath}.")

    return pTM, ipTM


def extract_average_pae(pae_json_filepath):
    average_pae = np.nan
    try:
        with open(pae_json_filepath, 'r') as f_pae:
            pae_data = json.load(f_pae)
    except Exception as e:
        print(f"Error reading PAE JSON file {pae_json_filepath}: {e}")
        return average_pae

    if isinstance(pae_data, dict):
        if 'predicted_aligned_error' in pae_data:
            pae_matrix = pae_data['predicted_aligned_error']
            if isinstance(pae_matrix, list) and all(isinstance(row, list) for row in pae_matrix):
                pae_matrix = np.array(pae_matrix)
                average_pae = np.mean(pae_matrix)
        else:
            print(f"Warning: 'predicted_aligned_error' key not found.")
    elif isinstance(pae_data, list) and len(pae_data) > 0:
        first_element = pae_data[0]
        if isinstance(first_element, dict) and 'predicted_aligned_error' in first_element:
            pae_matrix = first_element['predicted_aligned_error']
            if isinstance(pae_matrix, list) and all(isinstance(row, list) for row in pae_matrix):
                pae_matrix = np.array(pae_matrix)
                average_pae = np.mean(pae_matrix)
        else:
            print(f"Warning: 'predicted_aligned_error' key not found in first element.")
    else:
        print(f"Warning: PAE JSON has an unexpected structure.")

    return average_pae


def calculate_clash_score(pdb_filepath, distance_threshold=2.0):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('structure', pdb_filepath)
    except Exception as e:
        print(f"Error parsing PDB file {pdb_filepath}: {e}")
        return np.nan

    atom_coords = []
    residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element != 'H':
                        atom_coords.append(atom.get_coord())
                        residues.append(residue.get_id())

    if len(atom_coords) < 2:
        print(f"Insufficient atoms in {pdb_filepath} to calculate Clash Score.")
        return np.nan

    atom_coords = np.array(atom_coords)
    tree = cKDTree(atom_coords)
    pairs = tree.query_pairs(r=distance_threshold)
    total_clashes = 0

    for i, j in pairs:
        # skip if it's the same residue
        if residues[i][1] == residues[j][1]:
            continue
        total_clashes += 1

    # Return the actual number of clashes (NO percentage).
    return total_clashes


def run_pipeline(args):
    """
    Main pipeline to gather pLDDT, RMSD, pTM/ipTM, PAE, and Clash metrics,
    and then merge them into a single TSV. Also copies top 20 PDB files by pLDDT.
    """

    # Use user-specified or default values
    rf_base_dir  = args.rf_base_dir
    af_base_dir  = args.af_base_dir
    output_dir   = args.output_dir

    # Filenames for TSV outputs
    plddt_tsv       = os.path.join(output_dir, args.plddt_tsv)
    rmsd_tsv        = os.path.join(output_dir, args.rmsd_tsv)
    ptm_iptm_tsv    = os.path.join(output_dir, args.ptm_iptm_tsv)
    pae_tsv         = os.path.join(output_dir, args.pae_tsv)
    clash_tsv       = os.path.join(output_dir, args.clash_tsv)
    final_merged_tsv = os.path.join(output_dir, args.final_merged_tsv)

    # Folder to store the top 20 PDBs
    downselected_dir = os.path.join(output_dir, args.downselected_dir)

    os.makedirs(output_dir, exist_ok=True)

    # 1) pLDDT
    print("Collecting pLDDT scores from ranking_debug.json...")
    all_plddt_entries = []
    for root, dirs, files in os.walk(af_base_dir):
        for f in files:
            if f == "ranking_debug.json":
                json_path = os.path.join(root, f)
                results = extract_plddt_scores(json_path)
                all_plddt_entries.extend(results)

    plddt_df = pd.DataFrame(all_plddt_entries, columns=[
        "Name", "Object_Name", "Model_Name", "Ranked_File", "pLDDT_Score"
    ])
    plddt_df["Rank"] = plddt_df["Ranked_File"].str.extract(r"ranked_(\d+)\.pdb").astype(int)
    plddt_df.sort_values(by=["pLDDT_Score"], ascending=False, inplace=True)
    plddt_df.to_csv(plddt_tsv, sep="\t", index=False)
    print(f"pLDDT scores written to {plddt_tsv}")

    # 2) RMSD
    print("Calculating RMSD for reference PDBs vs. ranked_*.pdb ...")
    rmsd_results = []
    for ref_file in os.listdir(rf_base_dir):
        if not ref_file.endswith(".pdb"):
            continue
        ref_filepath = os.path.join(rf_base_dir, ref_file)
        ref_name = os.path.splitext(ref_file)[0]
        ref_identifier = ref_name.lower()

        print(f"\nProcessing reference PDB: {ref_filepath}")
        for root, dirs, files in os.walk(af_base_dir):
            relative_path = os.path.relpath(root, af_base_dir)
            top_level = relative_path.split(os.sep)[0].lower()
            if top_level != ref_identifier:
                continue

            for af_pdb_file in files:
                if af_pdb_file.startswith("ranked_") and af_pdb_file.endswith(".pdb"):
                    af_filepath = os.path.join(root, af_pdb_file)
                    try:
                        rank_str = af_pdb_file.split("_")[1].split(".")[0]
                        rank = int(rank_str)
                    except:
                        rank = -1

                    folder_two_up = os.path.basename(os.path.dirname(root))
                    name_col = folder_two_up + "/" + os.path.basename(root)
                    model_name = os.path.splitext(af_pdb_file)[0]
                    print(f"  AF PDB: {af_filepath} (Rank {rank})")

                    this_rmsd = calculate_rmsd(ref_filepath, af_filepath)
                    af_output_dir = os.path.basename(root)

                    rmsd_results.append((
                        name_col,
                        ref_name,
                        model_name,
                        rank,
                        this_rmsd,
                        af_output_dir
                    ))

    rmsd_df = pd.DataFrame(rmsd_results, columns=[
        "Name", "Reference", "Model_Name", "Rank", "RMSD", "AF_Output_Directory"
    ])
    rmsd_df.to_csv(rmsd_tsv, sep="\t", index=False)
    print(f"RMSD results saved to {rmsd_tsv}")

    # 3) pTM/ipTM, PAE, Clash
    print("Gathering pTM/ipTM, PAE, and Clash scores...")
    ptm_iptm_results = []
    pae_results = []
    clash_results = []

    for root, dirs, files in os.walk(af_base_dir):
        dirs[:] = [d for d in dirs if d != 'msas']  # skip 'msas' folder
        for file in files:
            if file.startswith("ranked_") and file.endswith(".pdb"):
                pdb_path = os.path.join(root, file)
                af_pdb_name = os.path.splitext(file)[0]
                object_name = os.path.basename(root)
                folder_two_up = os.path.basename(os.path.dirname(root))
                name_col = folder_two_up + "/" + object_name

                try:
                    rank_number = int(af_pdb_name.split('_')[1])
                except:
                    continue

                model_number = rank_number + 1
                model_identifier = f"model_{model_number}_multimer_v3_pred_0"

                pae_json_filename = f"pae_model_{model_number}_multimer_v3_pred_0.json"
                pae_json_filepath = os.path.join(root, pae_json_filename)
                ranking_debug_json = os.path.join(root, "ranking_debug.json")

                if not os.path.exists(ranking_debug_json):
                    continue
                if not os.path.exists(pae_json_filepath):
                    continue

                pTM_val, ipTM_val = extract_ptm_iptm(ranking_debug_json, model_identifier)
                avg_pae = extract_average_pae(pae_json_filepath)
                clash_val = calculate_clash_score(pdb_path, distance_threshold=2.0)

                ptm_iptm_results.append({
                    "Name": name_col,
                    "Object_Name": object_name,
                    "Model_Name": af_pdb_name,
                    "Rank": rank_number,
                    "pTM (%)": round(pTM_val, 2) if not math.isnan(pTM_val) else "N/A",
                    "ipTM (%)": round(ipTM_val, 2) if not math.isnan(ipTM_val) else "N/A"
                })
                pae_results.append({
                    "Name": name_col,
                    "Object_Name": object_name,
                    "Model_Name": af_pdb_name,
                    "Rank": rank_number,
                    "Average_PAE": round(avg_pae, 2) if not math.isnan(avg_pae) else "N/A"
                })
                clash_results.append({
                    "Name": name_col,
                    "Object_Name": object_name,
                    "Model_Name": af_pdb_name,
                    "Rank": rank_number,
                    # We now call this "Clashes" (no %) and store the actual clash count
                    "Clashes": round(clash_val, 2) if not math.isnan(clash_val) else "N/A"
                })

    ptm_iptm_df = pd.DataFrame(ptm_iptm_results)
    pae_df = pd.DataFrame(pae_results)
    clash_df = pd.DataFrame(clash_results)

    ptm_iptm_df.to_csv(ptm_iptm_tsv, sep="\t", index=False)
    pae_df.to_csv(pae_tsv, sep="\t", index=False)
    clash_df.to_csv(clash_tsv, sep="\t", index=False)

    print(f"pTM/ipTM saved to {ptm_iptm_tsv}")
    print(f"Average PAE saved to {pae_tsv}")
    print(f"Clash Scores (actual number of clashes) saved to {clash_tsv}")

    # 4) Merge ALL into one table
    print("Merging all metrics into one DataFrame...")

    # Merge plddt & RMSD 
    merged_all = plddt_df.merge(
        rmsd_df,
        on=["Name", "Rank"],
        how="left",
        suffixes=("", "_rmsd")
    )
    if "Object_Name_rmsd" in merged_all.columns:
        merged_all.drop(columns=["Object_Name_rmsd"], inplace=True)

    # Merge pTM/ipTM
    merged_all = merged_all.merge(
        ptm_iptm_df,
        on=["Name", "Rank"],
        how="left",
        suffixes=("", "_ptm")
    )
    if "Object_Name_ptm" in merged_all.columns:
        merged_all.drop(columns=["Object_Name_ptm"], inplace=True)

    # Merge PAE
    merged_all = merged_all.merge(
        pae_df,
        on=["Name", "Rank"],
        how="left",
        suffixes=("", "_pae")
    )
    if "Object_Name_pae" in merged_all.columns:
        merged_all.drop(columns=["Object_Name_pae"], inplace=True)

    # Merge Clash
    merged_all = merged_all.merge(
        clash_df,
        on=["Name", "Rank"],
        how="left",
        suffixes=("", "_clash")
    )
    if "Object_Name_clash" in merged_all.columns:
        merged_all.drop(columns=["Object_Name_clash"], inplace=True)

    merged_all.to_csv(final_merged_tsv, sep="\t", index=False)
    print(f"All metrics merged and written to {final_merged_tsv}")

    # Copy Top 20 PDB files by pLDDT
    print("Selecting and copying top 20 by pLDDT...")
    top_20_df = plddt_df.sort_values("pLDDT_Score", ascending=False).head(20)
    os.makedirs(downselected_dir, exist_ok=True)
    for idx, row in top_20_df.iterrows():
        name_val = row["Name"]
        ranked_file = row["Ranked_File"]
        src_pdb_path = os.path.join(af_base_dir, name_val, ranked_file)
        target_subdir = os.path.join(downselected_dir, name_val)
        os.makedirs(target_subdir, exist_ok=True)

        if os.path.isfile(src_pdb_path):
            shutil.copy(src_pdb_path, target_subdir)
        else:
            print(f"Warning: File not found - {src_pdb_path}")

    print(f"Top 20 PDB files copied into {downselected_dir}")
    print("\nPipeline done!\n")

    return {
        "af_base_dir": af_base_dir,
        "output_dir": output_dir,
        "plddt_tsv": plddt_tsv,
        "rmsd_tsv": rmsd_tsv,
        "clash_tsv": clash_tsv,
        "pae_tsv": pae_tsv,
        "ptm_iptm_tsv": ptm_iptm_tsv,
        "final_merged_tsv": final_merged_tsv
    }

# -----------------------------------------------------------------------
# The Report Code 
# -----------------------------------------------------------------------
def run_report(config, pdf_name):
   
    import matplotlib
    import sys
    import os
    import plotly.express as px
    import io

    plddt_file        = config["plddt_tsv"]
    rmsd_file         = config["rmsd_tsv"]
    clash_file        = config["clash_tsv"]
    pae_file          = config["pae_tsv"]
    ptm_file          = config["ptm_iptm_tsv"]
    final_merged_file = config["final_merged_tsv"]
    output_dir        = config["output_dir"]
    af_dir            = config["af_base_dir"]

    pdf_filename = os.path.join(output_dir, pdf_name)

    print("\n--- Starting PDF report generation ---\n")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory at {output_dir}")

    # Load pLDDT
    try:
        plddt_df = pd.read_csv(plddt_file, sep="\t")
        print("Successfully loaded pLDDT data.")
    except FileNotFoundError:
        print(f"Error: pLDDT file not found at {plddt_file}. Please check the path.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading pLDDT data: {e}")
        sys.exit(1)

    print("\n--- pLDDT DataFrame ---")
    print(plddt_df.head())

    # Possibly re-extract Rank
    if "Ranked_File" in plddt_df.columns:
        plddt_df['Rank'] = plddt_df['Ranked_File'].str.extract(r'ranked_(\d+)\.pdb').astype(int)

    max_plddt = plddt_df['pLDDT_Score'].max()
    if max_plddt <= 1.0:
        plddt_df['pLDDT_Score'] = plddt_df['pLDDT_Score'] * 100
        print("\nScaled pLDDT_Score from 0-1 to 0-100.")
    else:
        print("\npLDDT_Score is already scaled. No scaling applied.")

    # Load RMSD
    try:
        rmsd_df = pd.read_csv(rmsd_file, sep="\t")
        print("\nSuccessfully loaded RMSD data.")
    except FileNotFoundError:
        print(f"Error: RMSD file not found at {rmsd_file}. Please check the path.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading RMSD data: {e}")
        sys.exit(1)

    print("\n--- RMSD DataFrame ---")
    print(rmsd_df.head())

    # Merge pLDDT & RMSD
    merged_df = pd.merge(
        plddt_df,
        rmsd_df,
        on=["Name", "Rank"],
        how="inner"
    )

    plddt_keys = set(zip(plddt_df['Name'], plddt_df['Rank']))
    rmsd_keys = set(zip(rmsd_df['Name'], rmsd_df['Rank']))
    common_keys = plddt_keys.intersection(rmsd_keys)
    print(f"\nNumber of overlapping (Name, Rank) pairs: {len(common_keys)}")

    if len(common_keys) == 0:
        print("Error: No overlapping (Name, Rank) pairs found. Check data consistency.")
        sys.exit(1)
    else:
        print("Proceeding with merging based on overlapping keys.")

    print("\n--- Merged DataFrame ---")
    print(merged_df.head())
    print(f"Total merged rows: {len(merged_df)}")

    if merged_df.empty:
        print("\nError: Merged DataFrame is empty. Check if 'Name' and 'Rank' exist in both TSVs and align.")
        sys.exit(1)

    # Drop unneeded columns
    for col in ["Reference", "AF_Output_Directory", "Object_Name"]:
        if col in merged_df.columns:
            merged_df.drop(columns=[col], inplace=True)
            print(f"Dropped column: {col}")

    # Merge additional columns
    merged_df.rename(columns={
        'Model_Name_x': 'pLDDT_Model_Name',
        'Model_Name_y': 'RMSD_Model_Name'
    }, inplace=True, errors='ignore')

    merged_df['pLDDT_Score'] = pd.to_numeric(merged_df['pLDDT_Score'], errors="coerce")
    merged_df['RMSD'] = pd.to_numeric(merged_df['RMSD'], errors="coerce")
    merged_df.dropna(subset=["pLDDT_Score", "RMSD"], inplace=True)

    print("\n--- Cleaned Merged DataFrame ---")
    print(merged_df.head())
    print(f"Total merged rows after cleaning: {len(merged_df)}")

    # Sort for top 20
    merged_df_sorted = merged_df.sort_values(by=["pLDDT_Score", "RMSD"], ascending=[False, True])
    top_20 = merged_df_sorted.head(20).reset_index(drop=True)
    print("\n--- Top 20 Samples ---")
    print(top_20)

    if len(top_20) < 20:
        print(f"Warning: Only {len(top_20)} samples available for Top 20 selection.")

    print("\n--- Data Distribution ---")
    print(merged_df[['pLDDT_Score', 'RMSD']].describe())

    import plotly.express as px

    rank_colors = {
        '0': 'blue',
        '1': 'green',
        '2': 'orange',
        '3': 'red',
        '4': 'purple'
    }

    fig_all = px.scatter(
        merged_df,
        x="pLDDT_Score",
        y="RMSD",
        color=merged_df["Rank"].astype(str),
        color_discrete_map=rank_colors,
        hover_data={
            "Name": True,
            "pLDDT_Model_Name": True,
            "RMSD_Model_Name": True,
            "Rank": True,
            "pLDDT_Score": ":.2f",
            "RMSD": ":.2f"
        },
        labels={"pLDDT_Score": "pLDDT Score", "RMSD": "RMSD (Å)", "color": "Rank"},
        title="pLDDT vs RMSD Scatter Plot (All Samples)"
    )
    fig_all.update_traces(marker=dict(size=10, opacity=0.7, line=dict(width=1, color="DarkSlateGrey")))
    fig_all.update_layout(template="plotly_white")

    fig_top20 = px.scatter(
        top_20,
        x="pLDDT_Score",
        y="RMSD",
        color=top_20["Rank"].astype(str),
        color_discrete_map=rank_colors,
        hover_data={
            "Name": True,
            "pLDDT_Model_Name": True,
            "RMSD_Model_Name": True,
            "Rank": True,
            "pLDDT_Score": ":.2f",
            "RMSD": ":.2f"
        },
        labels={"pLDDT_Score": "pLDDT Score", "RMSD": "RMSD (Å)", "color": "Rank"},
        title="pLDDT vs RMSD Scatter Plot (Top 20 Samples)"
    )
    fig_top20.update_traces(marker=dict(size=12, opacity=0.9, line=dict(width=1, color="DarkSlateGrey")))
    fig_top20.update_layout(template="plotly_white")

    try:
        fig_all_png = fig_all.to_image(format="png", scale=2)
        fig_top20_png = fig_top20.to_image(format="png", scale=2)
        print("\nConverted Plotly figures to in-memory PNG bytes.")
    except Exception as e:
        print(f"Error converting figures to PNG: {e}")
        sys.exit(1)

    # Create table data for Top 20
    table_data = [["Standing", "Name", "pLDDT Model Name", "pLDDT Score", "RMSD (Å)"]]
    for idx, row in top_20.iterrows():
        table_data.append([
            idx + 1,
            row["Name"],
            row.get("pLDDT_Model_Name", "N/A"),
            f"{row['pLDDT_Score']:.2f}",
            f"{row['RMSD']:.2f}"
        ])

    # Merge the other metrics for a table
    try:
        clash_df = pd.read_csv(clash_file, sep="\t")
        print("\nSuccessfully loaded Clash_Score data.")

        pae_df = pd.read_csv(pae_file, sep="\t")
        print("Successfully loaded Average_PAE data.")

        ptm_df = pd.read_csv(ptm_file, sep="\t")
        print("Successfully loaded pTM and ipTM data.")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check the file paths.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading additional metrics data: {e}")
        sys.exit(1)

    try:
        metrics_merged_df = pd.merge(clash_df, pae_df, on=["Name", "Rank"], how="inner")
        metrics_merged_df = pd.merge(metrics_merged_df, ptm_df, on=["Name", "Rank"], how="inner")
        print("\nSuccessfully merged Clash_Score, Average_PAE, and pTM/ipTM data.")

        print("\n--- Merged Metrics DataFrame ---")
        print(metrics_merged_df.head())
        print(f"Total merged metrics rows: {len(metrics_merged_df)}")
    except Exception as e:
        print(f"Error merging metrics data: {e}")
        sys.exit(1)

    for col in ["Clashes", "Average_PAE", "pTM (%)", "ipTM (%)"]:
        if col in metrics_merged_df.columns:
            metrics_merged_df[col] = pd.to_numeric(metrics_merged_df[col], errors="coerce")
    metrics_merged_df.dropna(subset=["Clashes", "Average_PAE", "pTM (%)", "ipTM (%)"], inplace=True)

    # Adjusted table heading to say "Clashes" instead of "Clash Score (%)".
    metrics_table_data = [["Name", "Model Name", "Clashes", "Average PAE", "pTM (%)", "ipTM (%)"]]
    for _, row in metrics_merged_df.iterrows():
        model_val = row.get("Model_Name", "N/A")
        metrics_table_data.append([
            row["Name"],
            model_val,
            f"{row['Clashes']:.2f}",
            f"{row['Average_PAE']:.2f}",
            f"{row['pTM (%)']:.2f}",
            f"{row['ipTM (%)']:.2f}"
        ])

    # Build PDF
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
    from reportlab.platypus import Image as RLImage
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch

    try:
        doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
        elements = []

        styles = getSampleStyleSheet()
        title_style = styles["Title"]
        heading_style = styles["Heading2"]

        # 1) Title
        title = Paragraph(f"Analysis of ({af_dir})", title_style)
        elements.append(title)
        elements.append(Spacer(1, 12))

        # 2) First table (Top 20)
        table_heading = Paragraph("Top 20 Samples Ranked by pLDDT Score and RMSD", heading_style)
        elements.append(table_heading)
        elements.append(Spacer(1, 12))

        top20_table = Table(table_data, colWidths=[60, 140, 140, 80, 80])
        top20_table_style = TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4F81BD")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 12),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#D9E1F2")),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ])
        top20_table.setStyle(top20_table_style)
        elements.append(top20_table)
        elements.append(Spacer(1, 24))

        # 3) Scatter plots
        scatter_all_heading = Paragraph("Scatter Plot: All Samples", heading_style)
        elements.append(scatter_all_heading)
        elements.append(Spacer(1, 12))

        import io
        all_img_stream = io.BytesIO(fig_all_png)
        all_img = RLImage(all_img_stream, width=6.5*inch, height=4.5*inch)
        elements.append(all_img)
        elements.append(Spacer(1, 24))

        scatter_top20_heading = Paragraph("Scatter Plot: Top 20 Samples", heading_style)
        elements.append(scatter_top20_heading)
        elements.append(Spacer(1, 12))

        top20_img_stream = io.BytesIO(fig_top20_png)
        top20_img = RLImage(top20_img_stream, width=6.5*inch, height=4.5*inch)
        elements.append(top20_img)
        elements.append(Spacer(1, 24))

        # 4) Metrics table
        metrics_table_heading = Paragraph("Merged Metrics: Clashes, Average PAE, pTM, and ipTM", heading_style)
        elements.append(metrics_table_heading)
        elements.append(Spacer(1, 12))

        m_table = Table(metrics_table_data, colWidths=[120, 120, 80, 80, 70, 70])
        m_table_style = TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7030A0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 12),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#E2D4F0")),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ])
        m_table.setStyle(m_table_style)
        elements.append(m_table)
        elements.append(Spacer(1, 24))

        doc.build(elements)
        print(f"\nPDF report generated at: {pdf_filename}")

    except Exception as e:
        print(f"Error generating PDF report: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Pipeline to gather metrics and produce a PDF report.")
    parser.add_argument("--rf_base_dir",
                        help="Directory containing reference PDB files.")
    parser.add_argument("--af_base_dir",
                        help="Directory containing AlphaFold output subfolders.")
    parser.add_argument("--output_dir",
                        help="Directory to store TSVs, PDF, and subfolders.")

    # TSV Filenames
    parser.add_argument("--plddt_tsv",
                        help="Name of the pLDDT TSV file.")
    parser.add_argument("--rmsd_tsv",
                        help="Name of the RMSD TSV file.")
    parser.add_argument("--ptm_iptm_tsv",
                        help="Name of the pTM/ipTM TSV file.")
    parser.add_argument("--pae_tsv",
                        help="Name of the PAE TSV file.")
    parser.add_argument("--clash_tsv",
                        help="Name of the Clash TSV file.")
    parser.add_argument("--final_merged_tsv",
                        help="Name of the final merged TSV.")

    # Directory for top 20 PDB files
    parser.add_argument("--downselected_dir",
                        help="Name of the folder for top 20 PDB files inside --output_dir.")

    # PDF Name
    parser.add_argument("--pdf_name", default="analysis_report.pdf",
                        help="Name of the output PDF file.")

    args = parser.parse_args()

    # Run the pipeline
    config = run_pipeline(args)

    # Run the PDF generation
    run_report(config, pdf_name=args.pdf_name)


if __name__ == "__main__":
    main()
