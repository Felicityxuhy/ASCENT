import os
import copy
from turtle import pd
import numpy as np

from sklearn.cluster import KMeans
from torch_geometric.loader import DataLoader
from sklearn.metrics.pairwise import euclidean_distances

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from tqdm import tqdm
from sklearn.metrics import pairwise_distances

from utils import ActualSequentialSampler, test
from adapt.solvers.solver import get_solver


al_dict = {}

def register_strategy(name):
    def decorator(cls):
        al_dict[name] = cls
        return cls
    return decorator

def get_strategy(sample, *args):
	if sample not in al_dict: raise NotImplementedError
	return al_dict[sample](*args)

class SamplingStrategy:
	def __init__(self, dset, train_idx, model, discriminator, device, args, batch_size, balanced=False):
		self.dset = dset
		self.num_classes = 2
		self.train_idx = np.array(train_idx)
		self.model = model
		self.discriminator = discriminator
		self.device = device
		self.args = args
		self.idxs_lb = np.zeros(len(self.train_idx), dtype=bool)
		self.batch_size = batch_size

	def query(self, n):
		pass

	def update(self, idxs_lb):
		self.idxs_lb = idxs_lb

	def train(self, target_train_dset, target_val_loader, da_round=1, logger=None, src_loader=None, src_model=None, da_strat = None, al_strat = None,al_runs = None, ix = 0 , ix_all = 0):
		best_val_auc, best_model = 0.0, None

		da_save_dir = os.path.join('checkpoints', 'adapt', da_strat, al_strat + self.args.Suffix + al_runs )
		os.makedirs(da_save_dir, exist_ok=True) 
		if da_round>1:
			active_learning_model_last_round = os.path.join(da_save_dir,'Entropy_'+ da_strat +'_model_round{}.pth'.format(da_round-1))
			self.model.load_state_dict(torch.load(active_learning_model_last_round, map_location=torch.device('cpu')), strict=True)
			logger.info("查询后，第{}轮训练前加载上一轮最好的模型参数：{}".format(da_round, self.model.state_dict()['protein_embed.weight']))
			
		active_learning_model = os.path.join(da_save_dir,'Entropy_'+ da_strat +'_model_round{}.pth'.format(da_round))
	
		if os.path.exists(active_learning_model):
			self.model.load_state_dict(torch.load(active_learning_model, map_location=torch.device('cpu')), strict=True)
			best_model = copy.deepcopy(self.model)
		else:
			train_sampler = SubsetRandomSampler(self.train_idx[self.idxs_lb])
			tgt_sup_loader = DataLoader(target_train_dset, batch_size=self.batch_size, sampler=train_sampler, num_workers=0, drop_last=False)
			tgt_unsup_loader = DataLoader(target_train_dset, batch_size=self.batch_size, shuffle=False, num_workers=0, drop_last=False)
			
			opt_net_tgt = optim.Adam(self.model.parameters(), lr=self.args.adapt_lr,weight_decay=self.args.wd)

			early_stop = 0

			for epoch in range(self.args.adapt_num_epochs):
				if self.args.da_strat in ['dann','cdan'] :
					opt_dis_adapt = optim.Adam(self.discriminator.parameters(), lr=self.args.adapt_lr, \
											betas=(0.9, 0.999), weight_decay=0)
					solver = get_solver(self.args.da_strat, self.model, src_loader, tgt_sup_loader, tgt_unsup_loader, \
								self.train_idx, opt_net_tgt, da_round, self.device, self.args)
					solver.solve(epoch, self.discriminator, opt_dis_adapt, logger)

				elif self.args.da_strat in ['ft', 'mme','coral','lmmd', 'mmd', ]:
					solver = get_solver(self.args.da_strat, self.model, src_loader, tgt_sup_loader, tgt_unsup_loader, \
								self.train_idx, opt_net_tgt, da_round, self.device, self.args)
					solver.solve(epoch, logger)

				else:
					raise NotImplementedError
				
				for param_group in opt_net_tgt.param_groups:
						current_lr = param_group['lr']

				valid_Y, valid_P, valid_loss_a_epoch, valid_Accuracy, valid_Precision, valid_Recall, valid_AUC, valid_PRC,\
				valid_Sensitivity, valid_Specificity, valid_F1 = test(self.model, self.device, target_val_loader, split="val") 
				logger.info( f'[{epoch}/{self.args.adapt_num_epochs}] ' +
								'目标域性能' +
								f'valid_loss: {valid_loss_a_epoch:.5f} ' +
								f'valid_AUC: {valid_AUC:.5f} ' +
								f'valid_PRC: {valid_PRC:.5f} ' +
								f'valid_Accuracy: {valid_Accuracy:.5f} ' +
								f'valid_Precision: {valid_Precision:.5f} ' +
								f'valid_Sensitivity: {valid_Sensitivity:.5f} ' +
								f'valid_Specificity: {valid_Specificity:.5f} ' +
								f'valid_F1: {valid_F1:.5f} ' +
								f'valid_Reacll: {valid_Recall:.5f} ') 

				if (valid_AUC > best_val_auc):
					best_val_auc = valid_AUC
					best_model = copy.deepcopy(self.model)
					torch.save(best_model.state_dict(), active_learning_model)
					early_stop = 0
				else:
					early_stop += 1

				if early_stop >= self.args.early_stop_num :break

		return best_model, self.model

@register_strategy('Kmeans+margin')
class Kmeans_marginSampling1(SamplingStrategy):
	def __init__(self, dset, train_idx, model, discriminator, device, args, batch_size, balanced=False):
		super(Kmeans_marginSampling1, self).__init__(dset, train_idx, model, discriminator, device, args, batch_size)

	def query(self, n):
		if np.count_nonzero(self.idxs_lb) < self.args.total_budget * self.args.budget_spilt:
			idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]
			train_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])
			data_loader = DataLoader(self.dset, sampler=train_sampler, num_workers=0, batch_size=32, drop_last=False)
			self.model.eval()

			embeddings, all_scores = [], []
			with torch.no_grad():
				for batch_idx, data in tqdm(enumerate(data_loader)):
					data = data.to(self.device)
					labels = data.y
					labels = labels.to(self.device)
					features, predicted_scores, _ = self.model(data, self.device)
					embeddings.append(features)

			embeddings = torch.cat(embeddings) 
			embeddings = embeddings.cpu().numpy()
			embeddings = np.round(embeddings, 2)
			cluster_learner = KMeans(n_clusters=n, random_state=114514)
			cluster_learner.fit(embeddings)
			cluster_idxs = cluster_learner.predict(embeddings)  
			centers = cluster_learner.cluster_centers_[cluster_idxs]  
			dis = (embeddings - centers)**2 
			dis = dis.sum(axis=1)
			q_idxs = np.array([np.arange(embeddings.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])
			init_labeled = idxs_unlabeled[q_idxs]

			return init_labeled
		
		else:
			self.model.eval()
			idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]
			train_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])
			data_loader = DataLoader(self.dset, sampler=train_sampler, num_workers=0, batch_size=32, drop_last=False)
			
			all_log_probs, all_probs = [], []
			with torch.no_grad():
				for i, data in enumerate(tqdm(data_loader)):

					data = data.to(self.device)
					labels = data.y
					labels = labels.to(self.device)
					_, predicted_scores, _= self.model(data, self.device)
					probs = F.softmax(predicted_scores, 1).to('cpu')
					all_probs.append(probs)
			all_probs = torch.cat(all_probs)	
			probs_sorted, idxs = all_probs.sort(descending=True)	
			uncertainties = probs_sorted[:, 0] - probs_sorted[:,1]  

			return idxs_unlabeled[uncertainties.sort()[1][:n]]
