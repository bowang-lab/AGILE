from rdkit import Chem
from mordred import Calculator, descriptors
import pandas as pd

# List of new lipid SMILES (NRL1–NRL4)
new_smiles = [
    "NC(C1=CC=C[N+]([C@H]2[C@H](OC(CCCCCCCCCCCCCCCCC)=O)[C@H](OC(CCCCCCCCCCCCCCCCC)=O)[C@@H](CO)C2)=C1)=O",
    "NC(C1=CC=C[N+]([C@H]2[C@H](OC(CCCSSCCCCCCCCCCCCCC)=O)[C@H](OC(CCCSSCCCCCCCCCCCC)=O)[C@@H](CO)C2)=C1)=O",
    "NC(C1=CC=C[N+]([C@H]2[C@H](OC(CCCCCCC/C=C/CCCCCCCC)=O)[C@H](OC(CCCCCCC/C=C/CCCCCCCC)=O)[C@@H](CO)C2)=C1)=O",
    "NC(C1=CC=C[N+]([C@H]2[C@H](OC(CCCCCCCC=CC/C=C\CCCCC)=O)[C@H](OC(CCCCCCC/C=C/C/C=C\CCCCC)=O)[C@@H](CO)C2)=C1)=O"
]

# 1. Load one row of the candidate set template to get descriptor column specs
TEMPLATE_PATH = "data/candidate_set_smiles_plus_features.csv"
template = pd.read_csv(TEMPLATE_PATH, nrows=1)
all_cols = template.columns.tolist()
desc_cols = [c for c in all_cols if c.startswith("desc_")]
# Parse descriptor names and scaling factors from column names
desc_specs = []
for col in desc_cols:
    if '/' in col:
        base_name, scale = col.split('/')
        base_name = base_name.replace("desc_", "")
        scale = float(scale)
    else:
        base_name = col.replace("desc_", "")
        scale = 1.0
    desc_specs.append((col, base_name, scale))

# 2. Compute Mordred descriptors for the new SMILES
mols = [Chem.MolFromSmiles(s) for s in new_smiles]
calc = Calculator(descriptors, ignore_3D=True)
desc_df = calc.pandas(mols, nproc=1)
desc_df = desc_df.select_dtypes(['number']).fillna(0)

# 3. Scale/format descriptors to match AGILE columns
new_desc_data = pd.DataFrame(index=range(len(new_smiles)))
for col, base, scale in desc_specs:
    if base in desc_df.columns:
        new_desc_data[col] = desc_df[base] / scale
    else:
        new_desc_data[col] = 0.0
# Add the SMILES column
new_desc_data.insert(0, "smiles", new_smiles)

# 4. Append to the candidate set and save
candidates = pd.read_csv(TEMPLATE_PATH)
updated = pd.concat([candidates, new_desc_data], ignore_index=True)
updated.to_csv(TEMPLATE_PATH, index=False)
print(f"Added {len(new_smiles)} NRL structures to {TEMPLATE_PATH}") 