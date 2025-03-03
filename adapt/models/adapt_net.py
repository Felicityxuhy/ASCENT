import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init

class AdaptNet(nn.Module):
	"Defines an Adapt Network."
	def __init__(self, num_cls=2, model=None, src_weights_init=None, weights_init=None, weight_sharing='full', device=None, model_hp=None):
		super(AdaptNet, self).__init__()
		self.name = 'AdaptNet'
		self.cls_criterion = nn.CrossEntropyLoss()
		self.gan_criterion = nn.CrossEntropyLoss()
		self.weight_sharing = weight_sharing
		self.model_hp = model_hp
		self.device = device
		self.setup_net(model)


		if weights_init is not None:
			self.load(weights_init)
		elif src_weights_init is not None:
			self.load_src_net(src_weights_init)
		else:
			raise Exception('AdaptNet must be initialized with weights.')
	
	def custom_copy(self, src_net, weight_sharing):
		"""
		Vary degree of weight sharing between source and target CNN's
		"""
		tgt_net = copy.deepcopy(src_net)
		if weight_sharing != 'None':
			if weight_sharing == 'classifier': tgt_net.classifier = src_net.classifier
			elif weight_sharing == 'full': tgt_net = src_net
		return tgt_net
	
	def setup_net(self, model):
		"""Setup source, target and discriminator networks."""
		self.src_net = model
		self.tgt_net = self.custom_copy(self.src_net, self.weight_sharing)
		
		input_dim = 2
		self.discriminator = nn.Sequential(
				nn.Linear(input_dim, 500),
				nn.ReLU(),
				nn.Linear(500, 500),
				nn.ReLU(),
				nn.Linear(500, 2),
				)

	def load(self, init_path):
		"Loads full src and tgt models."
		net_init_dict = torch.load(init_path, map_location=torch.device('cpu'))
		self.load_state_dict(net_init_dict, strict=False)

	def load_src_net(self, init_path):
		"""Initialize source and target with source
		weights."""
		net_init_dict = torch.load(init_path, map_location=torch.device('cpu'))
		self.src_net.load_state_dict(net_init_dict, strict=False)
		self.tgt_net.load_state_dict(net_init_dict, strict=False)


	def save(self, out_path):
		torch.save(self.state_dict(), out_path)

	def save_tgt_net(self, out_path):
		torch.save(self.tgt_net.state_dict(), out_path)