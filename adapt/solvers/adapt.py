# -*- coding: utf-8 -*-
import sys
import utils
import random
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from tqdm.contrib import tzip
import torch.nn.functional as F
from .solver import register_solver
sys.path.append('../../')

class BaseSolver:
	"""
	Base DA solver class
	"""
	def __init__(self, net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, da_round, device, args):
		self.net = net
		self.src_loader = src_loader
		self.tgt_sup_loader = tgt_sup_loader
		self.tgt_unsup_loader = tgt_unsup_loader
		self.train_idx = np.array(train_idx)
		self.tgt_opt = tgt_opt
		self.da_round = da_round
		self.device = device
		self.args = args

	def solve(self, epoch):
		pass

@register_solver('ft')
class TargetFTSolver(BaseSolver):
	"""
	Finetune on target labels
	"""
	def __init__(self, net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, da_round, device, args):
		super(TargetFTSolver, self).__init__(net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, da_round, device, args)
	
	def solve(self, epoch, logger):
		self.net.train()		
		if self.da_round == 0:
			src_sup_wt, lambda_unsup = 1.0, 0.1
		else:
			src_sup_wt, lambda_unsup = self.args.src_sup_wt, self.args.unsup_wt
			tgt_sup_iter = iter(self.tgt_sup_loader)
		joint_loader = zip(self.src_loader, self.tgt_unsup_loader)
		# data_s源域训练数据   data_tu目标域未标注数据
		for batch_idx, ((data_s, data_tu)) in tqdm(enumerate(joint_loader)):	
			data_s = data_s.to(self.device)
			data_tu = data_tu.to(self.device)
			if self.da_round > 0:
				try:
					data_ts = next(tgt_sup_iter).to(self.device)
				except: break

			# zero gradients for optimizer
			self.tgt_opt.zero_grad()

			# extract features
			feature_s, score_s, _ = self.net(data_s, self.device)
			xeloss_src = src_sup_wt * nn.CrossEntropyLoss()(score_s,data_s.y.to(self.device))
			xeloss_tgt = 0
			if self.da_round > 0: 
				feature_ts, score_ts, _ = self.net(data_ts, self.device)
				xeloss_tgt = nn.CrossEntropyLoss()(score_ts, data_ts.y.to(self.device))
			xeloss = xeloss_src + xeloss_tgt 
			xeloss.backward()
			self.tgt_opt.step() 




@register_solver('mme')
class MMESolver(BaseSolver):
	"""
	Implements MME from Semi-supervised Domain Adaptation via Minimax Entropy: https://arxiv.org/abs/1904.06487
	"""
	def __init__(self, net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, da_round, device, args):
		super(MMESolver, self).__init__(net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, da_round, device, args)

	def solve(self, epoch, logger):
		"""
		Semisupervised adaptation via MME: XE on labeled source + XE on labeled target + \
										adversarial ent. minimization on unlabeled target
		"""
		self.net.train()		
		if self.da_round == 0:
			src_sup_wt = 1.0
		else:
			src_sup_wt = self.args.src_sup_wt
			tgt_sup_iter = iter(self.tgt_sup_loader)

		joint_loader = zip(self.src_loader, self.tgt_unsup_loader)
			
		#data_s源域训练数据   data_tu目标域未标注数据
		for batch_idx, ((data_s, data_tu)) in tqdm(enumerate(joint_loader)):	
			data_s = data_s.to(self.device)
			data_tu = data_tu.to(self.device)
			if self.da_round > 0:
				try:
					data_ts = next(tgt_sup_iter).to(self.device)
				except: break

			# zero gradients for optimizer
			self.tgt_opt.zero_grad()

			# extract features
			feature_s, score_s, _ = self.net(data_s, self.device)
			xeloss_src = src_sup_wt * nn.CrossEntropyLoss()(score_s,data_s.y.to(self.device))
			xeloss_tgt = 0
			if self.da_round > 0:
				feature_ts, score_ts, _ = self.net(data_ts, self.device)
				xeloss_tgt = nn.CrossEntropyLoss()(score_ts, data_ts.y.to(self.device))
			xeloss = xeloss_src + xeloss_tgt
			xeloss.backward()
			self.tgt_opt.step()

			feature_tu, score_tu, _ = self.net(data_tu, self.device , reverse_grad=True)  
			probs_tu = F.softmax(score_tu, dim=1)  
			loss_adent = self.args.mme_lambda * torch.mean(torch.sum(probs_tu * (torch.log(probs_tu + 1e-5)), 1))
			loss_adent.backward()
			self.tgt_opt.step() 

