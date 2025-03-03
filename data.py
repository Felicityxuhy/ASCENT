import os
import csv
import torch
import numpy as np
import networkx as nx
from   rdkit import Chem
from   torch_geometric import data as DATA
from   torch_geometric.data import InMemoryDataset, DataLoader
import torch.nn.functional as F


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    return c_size, features, edge_index


class ASDADataset:
    # Active Semi-supervised DA Dataset class
    def __init__(self, logger, seed, batch_size, num_workers, target, source_dataset_select,target_dataset_select):
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target = target
        self.source_dataset_select = source_dataset_select
        self.target_dataset_select = target_dataset_select
        
        # source domain data
        self.source_train_path = 'data/'+ self.source_dataset_select + '/train.csv'
        self.source_val_path = 'data/'+ self.source_dataset_select + '/val.csv'
        self.source_test_path = 'data/'+ self.source_dataset_select + '/test.csv'

        # target domain data
        self.target_train_path = 'data/'+ self.target_dataset_select + '/train.csv'
        self.target_val_path = 'data/'+ self.target_dataset_select + '/val.csv'
        self.target_test_path = 'data/'+ self.target_dataset_select + '/test.csv'

        self.src_dset = None
        self.train_size = None
        self.num_classes = None
        self.logger = logger
        self.num_workers = 0
        self.source_train_dataset = None
        self.source_valid_dataset = None
        self.source_test_dataset = None
        self.target_train_dataset = None
        self.target_valid_dataset = None
        self.target_test_dataset = None
        self.target_dset = None

    def get_dsets(self):
        self.source_valid_dataset = TestbedDataset(root='data', data_path = self.source_val_path, target = self.target, 
                                        data_part = 'valid', dataset_select =self.source_dataset_select, seed = self.seed)
        self.source_test_dataset = TestbedDataset(root='data', data_path = self.source_test_path, target = self.target, 
                                        data_part = 'test', dataset_select = self.source_dataset_select, seed = self.seed)
        self.source_train_dataset = TestbedDataset(root='data', data_path = self.source_train_path, target = self.target, 
                                        data_part = 'train', dataset_select = self.source_dataset_select, seed = self.seed)

        self.logger.info("源域 - 总数据集大小：{} ".format(len(self.source_train_dataset)+len(self.source_valid_dataset)+len(self.source_test_dataset)))

        return self.source_train_dataset,self.source_valid_dataset,self.source_test_dataset
    
    def get_loaders(self):
        if not self.source_train_dataset: self.get_dsets()
        train_idx = list(range(len(self.source_train_dataset)))

        train_loader = DataLoader(
            self.source_train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        val_loader = DataLoader(
            self.source_valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        test_loader = DataLoader(
            self.source_test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

        self.train_size = len(train_idx)
        return train_loader, val_loader, test_loader, train_idx, self.train_size

    def get_dsets_target(self):
        self.target_valid_dataset = TestbedDataset(root='data', data_path = self.target_val_path, target = self.target, 
                                        data_part = 'valid', dataset_select =self.target_dataset_select, seed = self.seed)
        self.target_test_dataset = TestbedDataset(root='data', data_path = self.target_test_path, target = self.target, 
                                        data_part = 'test', dataset_select = self.target_dataset_select, seed = self.seed)
        self.target_train_dataset = TestbedDataset(root='data', data_path = self.target_train_path, target = self.target, 
                                        data_part = 'train', dataset_select = self.target_dataset_select, seed = self.seed)

        self.logger.info("目标域 - 总数据集大小：{} ".format(len(self.target_train_dataset)+len(self.target_valid_dataset)+len(self.target_test_dataset)))
       
        return self.target_train_dataset,self.target_valid_dataset,self.target_test_dataset,
    
    def get_loaders_target(self):
        if not self.target_train_dataset: self.get_dsets_target()

        train_idx = list(range(len(self.target_train_dataset)))
        train_dataset = self.target_train_dataset
        train_loader = DataLoader(self.target_train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        val_loader = DataLoader(self.target_valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        test_loader = DataLoader(self.target_test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

        self.train_size = len(self.target_train_dataset)
        return train_loader, val_loader, test_loader, train_idx, self.train_size , train_dataset
    

def read_smiles_and_protein(data_path, target):
    smiles_data, labels = [], []
    protein_data = []
    protein_sequence = []
    with open(data_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            ID = i+1
            # get smiles
            smiles = row['SMILES']
            label = row[target]
            mol = Chem.MolFromSmiles(smiles)
            if mol != None and label != '':
                smiles_data.append(smiles)
            
            # get protein embedding
            protein_max = 1000
            proteinstr = row['Protein']
            path_parts = data_path.split('/')
            data_parts = path_parts[2].split('.')      
            protein_pt_path = 'data/esm_feature/{}/{}/ID_{}.pt'.format(path_parts[1],data_parts[0],str(ID))
            proteinint = torch.load(protein_pt_path)
            labels.append(int(label))
            protein_data.append(proteinint)
            protein_sequence.append(proteinstr)
                    
    print(len(smiles_data))
    print(len(protein_data))
    return smiles_data, protein_data, labels, protein_sequence


class TestbedDataset(InMemoryDataset):
    def __init__(self, data_path, target, data_part, dataset_select, seed, root='/tmp'):

        super(TestbedDataset, self).__init__(root)
        self.data_part = data_part
        self.seed = seed
        self.dataset_select = dataset_select
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.smiles_data, self.protein_embed, self.labels, protein_sequence= read_smiles_and_protein(data_path, target)
            self.process(self.smiles_data, self.protein_embed, self.labels, protein_sequence)
            self.data, self.slices = torch.load(self.processed_paths[0])
        
        self.conversion = 1

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset_select + '_' +self.data_part+'.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, xt, y, protein_sequences):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            print('saving features in pt file ^o^: {}/{}'.format(i+1, data_len))
            smiles = xd[i]  # smiles
            target = xt[i]  # 靶点
            labels = y[i]  # label
            protein_sequence = protein_sequences[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_to_graph(smiles)
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.LongTensor([labels]))
            
            target_feature = target['representations'][5]
            # padding
            target_shape = 1000
            pad_size = target_shape - target_feature.size(0)
            padding = (0, 0, 0, pad_size)

            # append graph, label and target sequence to data list
            GCNData.target = torch.FloatTensor([np.float32(F.pad(target_feature, padding))])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            GCNData.smiles = smiles
            GCNData.protein_sequence_str = protein_sequence
            data_list.append(GCNData)

        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

