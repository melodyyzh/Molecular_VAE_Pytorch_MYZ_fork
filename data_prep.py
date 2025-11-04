import pandas as pd
import argparse
from datasets import load_dataset

parser = argparse.ArgumentParser(description='Data Prep')
# parser.add_argument('file_loc', type=str, help='Path to the downloading dataset and name')
parser.add_argument('smiles_col_name', type=str, help='Name of the Column of SMILES')
parser.add_argument('save_path', type=str, help='Save Path of the Processed Data (CSV)')
parser.add_argument('len',type=int,help='Number of Training Points needed in Train and Val Data')
args = parser.parse_args()

print("Loading dataset 'n0w0f/qm9-csv' from Hugging Face...")
ds = load_dataset("n0w0f/qm9-csv")

# Combine all splits (e.g., 'train', 'validation', 'test') into one DataFrame
all_dfs = []
for split in ds.keys():
    print(f"Processing split: {split}")
    all_dfs.append(ds[split].to_pandas())

# 'chembl22' is now our combined DataFrame from all splits
chembl22 = pd.concat(all_dfs, ignore_index=True)
print("Dataset loaded and combined into a pandas DataFrame.")

# chembl22 = pd.read_csv(args.file_loc, sep=';', quotechar='"')

print("Available columns:")
print(chembl22.columns.tolist())
print(f"Looking for column: '{args.smiles_col_name}'")

# Add a check to make sure the column exists
if args.smiles_col_name not in chembl22.columns:
    print(f"Error: Column '{args.smiles_col_name}' not found!")
    print("Please check the available columns and try again.")
    exit()

print(chembl22[args.smiles_col_name][0:10])

df = pd.DataFrame({'SMILES':chembl22[args.smiles_col_name]})

# data cleaning
print(f"Original dataset size: {len(df)}")
df = df[df['SMILES'].notna()]  # Remove actual NaN values
df = df[df['SMILES'].astype(str).str.strip() != '']  # Remove empty strings
print(f"Dataset size after removing empty entries: {len(df)}")

df = df.reset_index(drop=True)
print(f"Sample of cleaned SMILES data:")
print(df['SMILES'].head(10))

df.to_csv(args.save_path+'.csv')

#Shuffling the Data to remove any sampling bias
df = df.sample(frac=1, random_state=42).reset_index(drop=True)


train_data = df.head(args.len)
val_data = df.tail(args.len)

train_data.to_csv(args.save_path+'_train.csv')
val_data.to_csv(args.save_path+'_val.csv')

print(f"Created files: {args.save_path}_train.csv and {args.save_path}_val.csv with {args.len} samples each")