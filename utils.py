import os
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from adapt.models.adapt_net import AdaptNet
from adapt.solvers.solver import get_solver
from torch.autograd import Function
from config.hyperparameter_model import hyperparameter
from torch.utils.data.sampler import Sampler
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, precision_score, recall_score,precision_recall_curve, auc, confusion_matrix, f1_score, roc_curve


class objectview(object):
	def __init__(self, d):
		self.__dict__ = d


class ActualSequentialSampler(Sampler):
	r"""Samples elements sequentially, always in the same order.

	Arguments:
		data_source (Dataset): dataset to sample from
	"""

	def __init__(self, data_source):
		self.data_source = data_source
        

	def __iter__(self):
		return iter(self.data_source)

	def __len__(self):
		return len(self.data_source)


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


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def train(model, device, train_loader, optimizer, scheduler, epoch, logger):
    """
    Test model on provided data for single epoch
    """
    train_losses_a_epoch = 0.0
    model.train()
    for trian_i, train_data in enumerate(tqdm(train_loader)):
        '''data preparation '''
        train_data = train_data.to(device)
        optimizer.zero_grad()
        
        _, predicted_interaction, _ = model(train_data, device)
        train_loss = nn.CrossEntropyLoss()(predicted_interaction, train_data.y.to(device))
        train_loss.backward()
        train_losses_a_epoch += train_loss.item()

        optimizer.step()
        scheduler.step()
    train_losses_a_epoch  /= len(train_loader)
    return train_losses_a_epoch


def test(model, device, test_loader, split="test", save_name = None, get_feature = False):
    """
    Test model on provided data
    """
    print('\nEvaluating model on {}...'.format(split))
    model.eval()
    test_losses = []
    test_losses_a_epoch = 0.0
    ATT = []
    protein, smiles = [], []
    Y, P, S = [], [], []
    origin_s = []
    features = []
    with torch.no_grad():
        for i, test_data in enumerate(tqdm(test_loader, mininterval=1e-9)):
            
            test_data = test_data.to(device)
            feature, predicted_scores, crossatt_maps = model(test_data, device)
            
            loss = nn.CrossEntropyLoss()(predicted_scores, test_data.y.to(device))
            correct_labels = test_data.y.to('cpu')

            predicted_scores_softmax = np.round(F.softmax(predicted_scores, dim=1).to('cpu').data.numpy(), 5)
            predicted_labels = np.argmax(predicted_scores_softmax, axis=1)
            predicted_scores_softmax = predicted_scores_softmax[:, 1]
            predicted_scores_origin = np.round(predicted_scores[:,1].to('cpu').data.numpy(),5)

            test_losses_a_epoch += loss.item()
            Y.extend(correct_labels)
            P.extend(predicted_labels)
            S.extend(predicted_scores_softmax)
            origin_s.extend(predicted_scores_origin)
            features.extend(feature.to('cpu').numpy())
            protein.extend(test_data.protein_sequence_str)
            smiles.extend(test_data.smiles)
            for j in range(crossatt_maps.size(0)):  # 遍历批次
                ATT.append(crossatt_maps[j])

    Precision = precision_score(Y, P)
    AUC = roc_auc_score(Y, S)
    precision, recall, _ = precision_recall_curve(Y, S)
    PRC = auc(recall, precision)
    Accuracy = accuracy_score(Y, P)
    # test_loss = np.average(test_losses)  
    test_losses_a_epoch /= len(test_loader)
    cm1 = confusion_matrix(Y, P)
    sensitivity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    specificity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    F1 = f1_score(Y, P)
    if get_feature:
        return features, Y, P, test_losses_a_epoch, Accuracy, Precision, sensitivity, AUC, PRC, sensitivity, specificity, F1
    else:   
        return Y, P, test_losses_a_epoch, Accuracy, Precision, sensitivity, AUC, PRC, sensitivity, specificity, F1

# UDA
def run_unsupervised_da(model, src_train_loader, tgt_sup_loader, tgt_unsup_loader,  src_test_loader, target_val_loader, tgt_test_loader, train_idx, device, name , args, model_hp, logger,time):
    """
    Unsupervised adaptation of source model to target at round 0
    Returns:
        Model post adaptation
    """
    cfg = hyperparameter()
    adapt_net_file = os.path.join('checkpoints', 'adapt', args.da_strat ,args.da_strat +'_model_fromsource_lr'+ str(args.adapt_lr_firstround) + name + '_' + args.Suffix + '.pth')
    if os.path.exists(adapt_net_file):
        print('Found pretrained checkpoint, loading...')
        adapt_model = AdaptNet(model = model, weights_init=adapt_net_file, device=device, model_hp=model_hp).to(device)
        best_model = adapt_model
    else:
        print('No pretrained checkpoint found, training...')
        source_path = f'checkpoints/source/{args.source_dataset_select}_source_{args.Suffix}.pth'
        adapt_model = AdaptNet(model = model, src_weights_init=source_path, device=device, model_hp=model_hp).to(device)
        opt_net_tgt = optim.Adam(adapt_model.tgt_net.parameters(), lr=float(args.adapt_lr_firstround)) #域适应优化器学习率，原来是1e-4

        best_val_auc = 0.0
        early_stop = 0
        for epoch in range(args.uda_num_epochs):
			
            pre_test_Y, pre_test_P, pre_test_loss_a_epoch, pre_test_Accuracy, pre_test_Precision, pre_test_Recall, pre_test_AUC, pre_test_PRC, \
			pre_test_Sensitivity, pre_test_Specificity, pre_test_F1 = test(adapt_model.tgt_net, device, tgt_test_loader, split="target_test") 
            logger.info(    '目标域测试集性能' +
                            f'本轮域适应前test_loss: {pre_test_loss_a_epoch:.5f} ' +
                            f'本轮域适应前test_AUC: {pre_test_AUC:.5f} ' +
                            f'本轮域适应前test_PRC: {pre_test_PRC:.5f} ' +
                            f'本轮域适应前test_Accuracy: {pre_test_Accuracy:.5f} ' +
                            f'本轮域适应前test_Precision: {pre_test_Precision:.5f} ' +
                            f'本轮域适应前test_Sensitivity: {pre_test_Sensitivity:.5f} ' +
                            f'本轮域适应前test_Specificity: {pre_test_Specificity:.5f} ' +
                            f'本轮域适应前test_F1: {pre_test_F1:.5f} ' +
                            f'本轮域适应前test_Reacll: {pre_test_Recall:.5f} ') 
            
            if args.da_strat in ['mme', 'ft']:
                uda_solver = get_solver(args.da_strat, adapt_model.tgt_net, src_train_loader, tgt_sup_loader, tgt_unsup_loader, \
                            train_idx, opt_net_tgt, 0, device, args)
                uda_solver.solve(epoch, logger)
				
            # valid Performance on the source domain 
            valid_src_Y, valid_arc_P, valid_src_loss_a_epoch, valid_src_Accuracy, valid_src_Precision, valid_src_Recall, valid_src_AUC, valid_src_PRC, valid_src_Sensitivity, valid_src_Specificity, valid_src_F1 = test(adapt_model.tgt_net, device, src_test_loader, split="val") 
            logger.info( f'[{epoch}/{args.uda_num_epochs}] ' +
                            '源域性能' +
                            f'valid_src_loss: {valid_src_loss_a_epoch:.5f} ' +
                            f'valid_src_AUC: {valid_src_AUC:.5f} ' +
                            f'valid_src_PRC: {valid_src_PRC:.5f} ' +
                            f'valid_src_Accuracy: {valid_src_Accuracy:.5f} ' +
                            f'valid_src_Precision: {valid_src_Precision:.5f} ' +
                            f'valid_src_Sensitivity: {valid_src_Sensitivity:.5f} ' +
                            f'valid_src_Specificity: {valid_src_Specificity:.5f} ' +
                            f'valid_src_F1: {valid_src_F1:.5f} ' +
                            f'valid_src_Reacll: {valid_src_Recall:.5f} ') 

            # valid Performance on the target domain
            valid_Y, valid_P, valid_loss_a_epoch, valid_Accuracy, valid_Precision, valid_Recall, valid_AUC, valid_PRC, valid_Sensitivity, valid_Specificity, valid_F1 = test(adapt_model.tgt_net, device, target_val_loader, split="val") 
            logger.info( f'[{epoch}/{args.uda_num_epochs}] ' +
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

            # eraly stop
            if (valid_AUC > best_val_auc):
                best_val_auc = valid_AUC
                best_model = adapt_model
                torch.save(best_model.state_dict(), adapt_net_file)
                early_stop = 0
            else:
                early_stop += 1

            # test Performance on the target domain
            test_Y, test_P, test_loss_a_epoch, test_Accuracy, test_Precision, test_Recall, test_AUC, test_PRC, test_Sensitivity, test_Specificity, test_F1 = test(adapt_model.tgt_net, device, tgt_test_loader, split="target_test") 
            logger.info(    '目标域测试集性能' +
                            f'test_loss: {test_loss_a_epoch:.5f} ' +
                            f'test_AUC: {test_AUC:.5f} ' +
                            f'test_PRC: {test_PRC:.5f} ' +
                            f'test_Accuracy: {test_Accuracy:.5f} ' +
                            f'test_Precision: {test_Precision:.5f} ' +
                            f'test_Sensitivity: {test_Sensitivity:.5f} ' +
                            f'test_Specificity: {test_Specificity:.5f} ' +
                            f'test_F1: {test_F1:.5f} ' +
                            f'test_Reacll: {test_Recall:.5f} ') 

    best_model = adapt_model
    net_init_dict = torch.load(adapt_net_file, map_location=torch.device('cpu'))
    best_model.load_state_dict(net_init_dict, strict=True)
    best_tgt_model, src_model, discriminator = best_model.tgt_net, adapt_model.src_net, adapt_model.discriminator

    return best_tgt_model, src_model, discriminator, adapt_net_file