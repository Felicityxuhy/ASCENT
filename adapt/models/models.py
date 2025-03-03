import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from adapt.mcan.mca import MCA_ED
from config.hyperparameter_model import mcan_cfg
from torch.autograd import Function

def binary_search(nums, target, flag): 
    left, right, index = 0, len(nums) - 1, len(nums)
    while left <= right:
        mid = (left + right) // 2
        if (nums[mid] > target) or (flag and nums[mid] >= target):
            right = mid - 1
            index = mid
        else:
            left = mid + 1
    return index

def make_mask(feature):
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)

class ReverseLayerF(Function):
	"""
	Gradient negation utility class
	"""				 
	@staticmethod
	def forward(ctx, x):
		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		output = grad_output.neg()
		return output, None
     
class ASCENT_Attention(nn.Module):
    def __init__(self, hp, seed, protein_MAX_LENGH = 1000, drug_MAX_LENGH = 100, heads = 2): 
        super(ASCENT_Attention, self).__init__()

        self.protein_dim = hp.protein_char_dim
        self.drug_dim = hp.drug_char_dim
        self.conv = hp.conv
        self.drug_MAX_LENGH = drug_MAX_LENGH
        self.drug_kernel = hp.drug_kernel
        self.protein_MAX_LENGH = protein_MAX_LENGH
        self.protein_kernel = hp.protein_kernel
        self.num_features_xd = hp.num_features_xd
        self.hidden_dims = 160 
        self.heads = heads
        
        self.__C = mcan_cfg()
        self.mcanatt = MCA_ED(self.__C)
        self.conv1 = GATConv(self.num_features_xd, self.hidden_dims//self.heads, heads=self.heads)
        self.conv2 = GATConv(self.hidden_dims, self.hidden_dims//self.heads, heads=self.heads)
        self.protein_embed = nn.Embedding(26, self.protein_dim, padding_idx=0)
        self.Drug_max_pool = nn.MaxPool1d(self.drug_MAX_LENGH)
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.protein_dim, out_channels=self.conv, kernel_size= self.protein_kernel[0] , padding=3),
            nn.ReLU(),
        )

        self.Protein_max_pool = nn.MaxPool1d(self.protein_MAX_LENGH)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(320, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)
        self.printnum = 1

    def forward(self, data, device, reverse_grad=False):

        drug, drug_edge_index, drug_batch = data.x, data.edge_index, data.batch
        protein_feature = data.target
        batch_size = len(data.y)

        # drug GAT embedding 
        x = self.conv1(drug, drug_edge_index)
        x = self.relu(x)
        x = self.conv2(x, drug_edge_index)
        x = self.relu(x)
        
        full_matrix = x
        embedding_split = torch.zeros((len(torch.unique(data.batch)),100, self.hidden_dims))
        data_batch_index = []
        for i in range(len(torch.unique(data.batch))):
            data_batch_index.append(binary_search(data.batch, i, True))

        # drug max_length 100 / Zero-padding
        for i in range(len(torch.unique(data.batch))):
            if i > 0 :
                single_mol_embedding = full_matrix[data_batch_index[i-1]:data_batch_index[i],:]
                if single_mol_embedding.shape[0]<=100 :                    
                    pad_zeros = torch.zeros((100-single_mol_embedding.shape[0], self.hidden_dims)).to(device)
                    embedding_split[i-1] = torch.cat((single_mol_embedding, pad_zeros),dim=0)
                else:
                    embedding_split[i-1] = single_mol_embedding[:100]
    
        drugembed = embedding_split.to(device)  
        drugembed_mask = make_mask(drugembed)  
        proteinembed_mask = make_mask((protein_feature)) 

        # protein Conv1d embedding
        protein_conv = self.Protein_CNNs(protein_feature.permute(0, 2, 1))
        protein_conv = protein_conv.permute(0, 2, 1)

        # cross attention
        macnatt_protein , macnatt_drug, crossatt_maps = self.mcanatt(protein_conv,drugembed,proteinembed_mask,drugembed_mask)

        # get attention distribution maps
        crossatt_maps = torch.mean(crossatt_maps, dim=1) 
        crossatt_maps = torch.mean(crossatt_maps, dim=1)

        macnatt_drug_pool = self.Drug_max_pool(macnatt_drug.permute(0, 2, 1)).squeeze(2)
        macnatt_protein_pool = self.Protein_max_pool(macnatt_protein.permute(0, 2, 1)).squeeze(2)

        pair = torch.cat([macnatt_protein_pool, macnatt_drug_pool], dim=1)
        if reverse_grad: pair = ReverseLayerF.apply(pair)
        output_feature = pair

        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        predict = self.out(fully2)

        return output_feature, predict, crossatt_maps

