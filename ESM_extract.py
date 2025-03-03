import csv
import os
import argparse

parser = argparse.ArgumentParser()	
parser.add_argument('--DATASET', type=str, default="bindingdb")
args = parser.parse_args()

DATASET = args.DATASET
# 输出的FASTA文件路径
path = [ 
[f'data/{DATASET}/train.csv', f'data/fasta_data/{DATASET}/train'],
[f'data/{DATASET}/val.csv', f'data/fasta_data/{DATASET}/val'],
[f'data/{DATASET}/test.csv', f'data/fasta_data/{DATASET}/test']
]
for data_path , save_path in path :
    with open(data_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = save_path + '/data.fasta'
        with open(save_path, mode='w', encoding='utf-8') as fasta_file:
            for i, row in enumerate(csv_reader):
                # Using row indices as protein IDs
                protein_id = f"ID_{i+1}"
                protein_sequence = row['Protein']
                fasta_file.write(f">{protein_id}\n")
                fasta_file.write(f"{protein_sequence}\n")
