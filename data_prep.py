import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Data Prep')
parser.add_argument('file_loc', type=str, help='Path to the downloading dataset and name')
parser.add_argument('smiles_col_name', type=str, help='Name of the Column of SMILES')
parser.add_argument('save_path', type=str, help='Save Path of the Processed Data (CSV)')
parser.add_argument('len',type=int,help='Number of Training Points needed in Train and Val Data')
args = parser.parse_args()

chembl22 = pd.read_csv(args.file_loc, sep=';', quotechar='"')

print("Available columns:")
print(chembl22.columns.tolist())
print(f"Looking for column: '{args.smiles_col_name}'")

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