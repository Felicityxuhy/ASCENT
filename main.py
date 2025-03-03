import csv
import os
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
import copy
import torch
import argparse
import distutils
import numpy as np
import torch.nn as nn
from distutils import util
import torch.optim as optim
from scipy.stats import norm
from tqdm.contrib import tzip
from tqdm import tqdm, trange
from omegaconf import OmegaConf
from sample import get_strategy
from collections import defaultdict
from data import ASDADataset
from sklearn.metrics.pairwise import cosine_similarity
from adapt.models.models import ASCENT_Attention
from config.hyperparameter_model import hyperparameter, mcan_cfg
from utils import get_logger, run_unsupervised_da, test, train


def run_active_adaptation(args, source_model, src_dset, device, logger, cfg): 
    cfg = hyperparameter()
    discriminator = None
    src_train_loader, _, src_test_loader, _, _ = src_dset.get_loaders()
    target_dset = ASDADataset(logger=logger, seed = cfg.seed, batch_size = cfg.batch_size, num_workers = cfg.num_workers, 
                              target = cfg.target, source_dataset_select = args.source_dataset_select, 
                                target_dataset_select = args.target_dataset_select)
    target_train_loader, target_val_loader, target_test_loader, train_idx, target_train_size, target_train_dset = target_dset.get_loaders_target()

    target_aucs = defaultdict(list)
    exp_name = '{}_{}_{}_{}runs_{}rounds_{}budget'.format(args.model_init, args.al_strat, args.da_strat, args.runs, args.num_rounds, args.total_budget)
    
    # Sample varying % of target data
    sampling_ratio = [(args.total_budget/args.num_rounds) * n for n in range(args.num_rounds+1)]
    DA_test_result_savepath = f'result/{args.source_dataset_select}_{args.da_strat}_{args.al_strat}{args.Suffix}'
    if not os.path.exists(DA_test_result_savepath):
            os.makedirs(DA_test_result_savepath)

    _, _, _, _, transfer_Accuracy, transfer_Precision, transfer_Reacll, transfer_AUC, transfer_PRC, transfer_Sensitivity, transfer_Specificity, transfer_F1 = test(source_model, device, target_test_loader, get_feature = True)
    out_str = '使用域适应算法前DTI performance (Before {}): AUC={:.5f}, PRC={:.5f}, ACC={:.5f}, PRECISION={:.5f}, Sensitivity={:.5f}, Specificity={:.5f}, F1={:.5f}, RECALL={:.5f}'.format(args.da_strat, transfer_AUC, transfer_PRC, transfer_Accuracy, transfer_Precision, transfer_Sensitivity, transfer_Specificity, transfer_F1, transfer_Reacll)
    logger.info(out_str)

    with open(DA_test_result_savepath + '/DA_results.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["DA前/后", "AUC", "PRC", "Accuracy", "Precision", "Sensitivity", "Specificity", "F1", "Recall"])
                writer.writerow([
                        f"DA前目标域", 
                        f"{transfer_AUC:.5f}", 
                        f"{transfer_PRC:.5f}", 
                        f"{transfer_Accuracy:.5f}", 
                        f"{transfer_Precision:.5f}", 
                        f"{transfer_Sensitivity:.5f}", 
                        f"{transfer_Specificity:.5f}", 
                        f"{transfer_F1:.5f}", 
                        f"{transfer_Reacll:.5f}"
                    ])                     

    logger.info('------------------------------------------------------\n')
    logger.info('Running strategy: Init={} AL={} Train={}'.format(args.model_init, args.al_strat, args.da_strat))
    logger.info('\n------------------------------------------------------')	

    model = source_model
    if args.da_strat != 'ft':
        print('Round 0: Unsupervised DA to target via {}'.format(args.da_strat))
        best_model, src_model, discriminator, adapt_net_file = run_unsupervised_da(model, src_train_loader, None, target_train_loader, \
                                                                    src_test_loader,target_val_loader, target_test_loader, train_idx, device, f'{args.source_dataset_select} to {args.target_dataset_select}' ,args, cfg, logger,time)
        _, _, _, _, source_Accuracy, source_Precision, source_Reacll, source_AUC, source_PRC, source_Sensitivity, source_Specificity, source_F1 = test(best_model, device, src_test_loader, get_feature = True)
        _, _, _, _, transfer_Accuracy, transfer_Precision, transfer_Reacll, transfer_AUC, transfer_PRC, transfer_Sensitivity, transfer_Specificity, transfer_F1 = test(best_model, device, target_test_loader, get_feature = True)

        out_target_str = '使用域适应算法后 目标域 DTI performance (After {}): AUC={:.5f}, PRC={:.5f}, ACC={:.5f}, PRECISION={:.5f}, Sensitivity={:.5f}, Specificity={:.5f}, F1={:.5f} ,RECALL={:.5f}'.format(args.da_strat, transfer_AUC, transfer_PRC, transfer_Accuracy, transfer_Precision, transfer_Sensitivity, transfer_Specificity,transfer_F1, transfer_Reacll)
        out_src_str = '使用域适应算法后 源域 DTI performance (After {}): AUC={:.5f}, PRC={:.5f}, ACC={:.5f}, PRECISION={:.5f}, Sensitivity={:.5f}, Specificity={:.5f}, F1={:.5f} ,RECALL={:.5f}'.format(args.da_strat, source_AUC, source_PRC, source_Accuracy, source_Precision, source_Sensitivity, source_Specificity, source_F1, source_Reacll)
        logger.info(out_target_str)
        logger.info(out_src_str)
        with open(DA_test_result_savepath + '/'+ args.al_runs + 'DA_results.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                            f"DA后目标域", 
                            f"{transfer_AUC:.5f}", 
                            f"{transfer_PRC:.5f}", 
                            f"{transfer_Accuracy:.5f}", 
                            f"{transfer_Precision:.5f}", 
                            f"{transfer_Sensitivity:.5f}", 
                            f"{transfer_Specificity:.5f}", 
                            f"{transfer_F1:.5f}", 
                            f"{transfer_Reacll:.5f}"
                        ])    

    else:
        print('Round 0: ft {}'.format(args.da_strat))
        best_model, src_model, discriminator, adapt_net_file = run_unsupervised_da(model, src_train_loader, None, target_train_loader, \
                                                                    src_test_loader,target_val_loader, target_test_loader, train_idx, device, f'{args.source_dataset_select} to {args.target_dataset_select}' , args, cfg, logger, time)
        _, _, _, source_Accuracy, source_Precision, source_Reacll, source_AUC, source_PRC, source_Sensitivity, source_Specificity, source_F1 = test(best_model, device, src_test_loader)
        _, _, _, transfer_Accuracy, transfer_Precision, transfer_Reacll, transfer_AUC, transfer_PRC, transfer_Sensitivity, transfer_Specificity, transfer_F1 = test(best_model, device, target_test_loader)
        out_target_str = 'ft 后 目标域 DTI performance (After {}): AUC={:.5f}, PRC={:.5f}, ACC={:.5f}, PRECISION={:.5f}, Sensitivity={:.5f}, Specificity={:.5f}, F1={:.5f}, RECALL={:.5f}'.format(args.da_strat, transfer_AUC, transfer_PRC, transfer_Accuracy, transfer_Precision, transfer_Sensitivity, transfer_Specificity, transfer_F1, transfer_Reacll)
        out_src_str = 'ft 后 源域 DTI performance (After {}): AUC={:.5f}, PRC={:.5f}, ACC={:.5f}, PRECISION={:.5f}, Sensitivity={:.5f}, Specificity={:.5f}, F1={:.5f}, RECALL={:.5f}'.format(args.da_strat, source_AUC, source_PRC, source_Accuracy, source_Precision, source_Sensitivity, source_Specificity, source_F1 ,source_Reacll)
        logger.info(out_target_str)
        logger.info(out_src_str)



    if args.task == 'ADA':
        tqdm_run = trange(args.runs)
        AL_selected_samples_path = 'checkpoints/adapt/' + args.da_strat + '/' + args.al_strat + args.Suffix + args.al_runs + '/'
        if not os.path.exists(AL_selected_samples_path):
            os.makedirs(AL_selected_samples_path)
        #将主动学习挑选的样本保存到csv
        with open(AL_selected_samples_path+'AL_selected_samples.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['SMILES','Protein','Y','ID'])


        #主动学习结果记录
        with open(AL_selected_samples_path + 'AL_results.csv', mode='w', newline='') as file:
                    writer = csv.writer(file)
                    # 写入表头
                    writer.writerow(["AL_ROUND", "AUC", "PRC", "Accuracy", "Precision", "Sensitivity", "Specificity", "F1", "Recall"])

        for run in tqdm_run: # Run over multiple experimental runs  实验次数
            tqdm_run.set_description('Run {}'.format(str(run)))
            tqdm_run.refresh()#刷新进度条
            tqdm_rat = trange(len(sampling_ratio[1:]))
            target_aucs[0.0].append(transfer_AUC)  # 第0轮的AUC值

            # 每次运行开始从最佳模型状态重新开始，确保独立性
            curr_model = copy.deepcopy(best_model)  # 如果是别的模型，记得改成best_model
            curr_source_model = curr_model
            logger.info('主动学习实验开始时候的模型参数：{}'.format(curr_model.state_dict()['protein_embed.weight']))

            # Keep track of labeled vs unlabeled data  训练集的所有样本的索引
            idxs_lb = np.zeros(len(train_idx), dtype=bool)  # 表示训练集中的哪些样本已经被标记（1）或未被标记（0），初始所有样本视为0
            # 初始化主动学习策略
            batch_size = cfg.batch_size
            pre_test_Y, pre_test_P, pre_test_loss_a_epoch, pre_test_Accuracy, pre_test_Precision, pre_test_Recall, pre_test_AUC, pre_test_PRC, \
            pre_test_Sensitivity, pre_test_Specificity, pre_test_F1 = test(curr_source_model, device, target_test_loader, split="target_test") 
            logger.info(    '本轮域适应前目标域测试集性能' +
                            f'test_loss: {pre_test_loss_a_epoch:.5f} ' +
                            f'test_AUC: {pre_test_AUC:.5f} ' +
                            f'test_PRC: {pre_test_PRC:.5f} ' +
                            f'test_Accuracy: {pre_test_Accuracy:.5f} ' +
                            f'test_Precision: {pre_test_Precision:.5f} ' +
                            f'test_Sensitivity: {pre_test_Sensitivity:.5f} ' +
                            f'test_Specificity: {pre_test_Specificity:.5f} ' +
                            f'test_F1: {pre_test_F1:.5f} ' +
                            f'test_Reacll: {pre_test_Recall:.5f} ') 
            sampling_strategy = get_strategy(args.al_strat, target_train_dset, train_idx, \
                                                curr_model, discriminator, device, args, batch_size)	
            
            for ix in tqdm_rat: # Iterate over Active DA rounds 主动学习迭代的轮次，采样数量循环
                ratio = sampling_ratio[ix+1]
                tqdm_rat.set_description('# Target labels={:d}'.format(int(ratio)))
                tqdm_rat.refresh()

                # Select instances via AL strategy
                logger.info('\nSelecting instances...')
                idxs = sampling_strategy.query(int(sampling_ratio[1])) # 选取样本
                #将主动学习挑选的样本保存到csv
                for idx in idxs:
                    with open(AL_selected_samples_path+'AL_selected_samples.csv', mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([target_train_dset[idx].smiles, target_train_dset[idx].protein_sequence_str, int(target_train_dset[idx].y), idx])
                idxs_lb[idxs] = True
                sampling_strategy.update(idxs_lb)

                # Update model with new data via DA strategy  根据DA策略更新模型
                logger.info('主动学习取样本之后，域适应之前还没加载最好模型的模型参数：{}'.format(curr_model.state_dict()['protein_embed.weight']))
                best_model, last_epoch_model = sampling_strategy.train(target_train_dset, target_val_loader, da_round=(ix+1), \
                                                        logger=logger, \
                                                        src_loader=src_train_loader, \
                                                        src_model=curr_source_model, da_strat = args.da_strat, al_strat = args.al_strat, al_runs = args.al_runs, ix = ix, ix_all = len(train_idx) )
                
                logger.info("第{}轮训练之后最后一轮的模型参数：{}".format((ix+1), last_epoch_model.state_dict()['protein_embed.weight']))
                logger.info("第{}轮训练之后最好的模型参数：{}".format((ix+1), best_model.state_dict()['protein_embed.weight']))

                # 测试在目标域测试集上的性能
                _, _, _, AL_transfer_Accuracy, AL_transfer_Precision, AL_transfer_Reacll, AL_transfer_AUC, AL_transfer_PRC,\
                AL_Sensitivity, AL_Specificity, AL_F1 = test(best_model, device, target_test_loader, save_name = str(ix+1))

                out_str = 'AL Test performance (Round {}, # Target labels={:d}): AUC={:.5f}, PRC={:.5f}, ACC={:.5f}, PRECISION={:.5f}, Sensitivity={:.5f}, Specificity={:.5f}, F1={:.5f}, RECALL={:.5f}'.format(ix+1, int(ratio), \
                        AL_transfer_AUC, AL_transfer_PRC, AL_transfer_Accuracy, AL_transfer_Precision, AL_Sensitivity, AL_Specificity, AL_F1, AL_transfer_Reacll)
                logger.info('\n------------------------------------------------------\n')
                logger.info(out_str)
                logger.info('域适应性能提升' + f'AUC提升: {AL_transfer_AUC - pre_test_AUC:.5f} ') 
                target_aucs[ratio].append(AL_transfer_AUC)
                with open(AL_selected_samples_path + '/AL_results.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                            f"{ix + 1:.1f}", 
                            f"{AL_transfer_AUC:.5f}", 
                            f"{AL_transfer_PRC:.5f}", 
                            f"{AL_transfer_Accuracy:.5f}", 
                            f"{AL_transfer_Precision:.5f}", 
                            f"{AL_Sensitivity:.5f}", 
                            f"{AL_Specificity:.5f}", 
                            f"{AL_F1:.5f}", 
                            f"{AL_transfer_Reacll:.5f}"
                        ])    

            wargs = vars(args) if isinstance(args, argparse.Namespace) else dict(args)
            target_aucs['args'] = wargs
            logger.info(exp_name)

def main(time):
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()	
    parser.add_argument('--load_from_cfg', type=lambda x:bool(distutils.util.strtobool(x)), default=True, help="Load from config?")
    parser.add_argument('--cfg_file', type=str, help="Experiment configuration file", default="config/mme.yml")
    parser.add_argument('--source_dataset_select', type=str, help="source domain data", default="test_dataset")
    parser.add_argument('--target_dataset_select', type=str, help="target domain data", default="test_dataset")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--al_runs', type=str, help="Number of repeated experiments", default="1")
    parser.add_argument('--Suffix', type=str, help="Define your comment content", default="")
    parser.add_argument('--al_strat', type=str, help="Sampling strategy", default="Kmeans+margin")   
    parser.add_argument('--task', type=str, help="None_DA / DA / ADA", default="None_DA")   

    args_cmd = parser.parse_args()
    cfg = hyperparameter()

    if args_cmd.load_from_cfg:
        args_cfg = dict(OmegaConf.load(args_cmd.cfg_file))
        args_cmd = vars(args_cmd)
        for k in args_cmd.keys():
            if args_cmd[k] is not None: args_cfg[k] = args_cmd[k]
        args = OmegaConf.create(args_cfg)
    else: 
        args = args_cmd

    # logger
    method = args.cfg_file.split("/")[1].split(".")[0]
    logger = get_logger(f'logs/{method}_{args.al_strat}{args.Suffix}_{args.al_runs}.log')
    logger.info(args)

    device = args.device    
    # load source domain dataset
    src_dset = ASDADataset(logger=logger, seed = cfg.seed, batch_size = cfg.batch_size, num_workers = cfg.num_workers, 
                              target = cfg.target, source_dataset_select = args.source_dataset_select, target_dataset_select = args.target_dataset_select)
    src_train_loader, src_val_loader, src_test_loader, _ ,src_train_size = src_dset.get_loaders()
    cfg = hyperparameter()
    MCAN_cfg = mcan_cfg()
    source_model = ASCENT_Attention(cfg, cfg.seed).to(device)

    # Initialize the source model
    weight_p, bias_p = [], []
    for p in source_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for name, p in source_model.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    source_file = f'{args.source_dataset_select}_source_{args.Suffix}.pth'
    source_path = os.path.join('checkpoints', 'source', source_file)

    if os.path.exists(source_path): 
        logger.info('Loading source checkpoint: {}'.format(source_path))
        source_model.load_state_dict(torch.load(source_path, map_location=device), strict=True)
        best_source_model = source_model
    else:	 
        best_epoch, best_val_auc, best_source_model = 0, 0.0, None
    
        source_optimizer = optim.AdamW(
            [{'params': weight_p, 'weight_decay': cfg.weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=cfg.Learning_rate)
        source_scheduler = optim.lr_scheduler.CyclicLR(source_optimizer, base_lr=cfg.Learning_rate, max_lr=cfg.Learning_rate*10, cycle_momentum=False,
                                                step_size_up=src_train_size // args.batch_size)
        epoch_len = len(str(cfg.Epoch))
        hp_attr = '\n'.join(['%s:%s' % item for item in cfg.__dict__.items()])
        MCAN_cfg = '\n'.join(['%s:%s' % item for item in MCAN_cfg.__dict__.items()])
        logger.info('模型超参数:{}'.format(hp_attr))
        logger.info('模型超参数:{}'.format(MCAN_cfg))
        for epoch in range(1, cfg.Epoch + 1):
            train_loss_a_epoch = train(source_model, device, src_train_loader, source_optimizer, source_scheduler, epoch, logger)
            valid_Y, valid_P, valid_loss_a_epoch, valid_Accuracy, valid_Precision, valid_Recall, valid_AUC, valid_PRC, valid_Sensitivity, valid_Specificity , valid_F1= test(source_model, device, src_val_loader, split="val") 
            _, _, test_loss_a_epoch_intrain, test_Accuracy_intrain, test_Precision_intrain, test_Recall_intrain, test_AUC_intrain, test_PRC_intrain, test_Sensitivity_intrain, test_Specificity_intrain , test_F1_intrain= test(source_model, device, src_test_loader, split="test") 

            logger.info( f'[{epoch:>{epoch_len}}/{cfg.Epoch:>{epoch_len}}] ' +
                            f'train_loss: {train_loss_a_epoch:.5f} ' +
                            f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                            f'valid_AUC: {valid_AUC:.5f} ' +
                            f'valid_PRC: {valid_PRC:.5f} ' +
                            f'valid_Accuracy: {valid_Accuracy:.5f} ' +
                            f'valid_Pre cision: {valid_Precision:.5f} ' +
                            f'valid_Sensitivity: {valid_Sensitivity:.5f} ' +
                            f'valid_Specificity: {valid_Specificity:.5f} ' +
                            f'valid_F1: {valid_F1:.5f} ' +
                            f'valid_Reacll: {valid_Recall:.5f} ') 
            
            logger.info( f'[{epoch:>{epoch_len}}/{cfg.Epoch:>{epoch_len}}] ' +
                            f'test_loss_a_epoch_intrain: {test_loss_a_epoch_intrain:.5f} ' +
                            f'test_AUC_intrain: {test_AUC_intrain:.5f} ' +
                            f'test_PRC_intrain: {test_PRC_intrain:.5f} ' +
                            f'test_Accuracy_intrain: {test_Accuracy_intrain:.5f} ' +
                            f'test_Precision_intrain cision: {test_Precision_intrain:.5f} ' +
                            f'test_Sensitivity_intrain: {test_Sensitivity_intrain:.5f} ' +
                            f'test_Specificity_intrain: {test_Specificity_intrain:.5f} ' +
                            f'test_F1_intrain: {test_F1_intrain:.5f} ' +
                            f'test_Recall_intrain: {test_Recall_intrain:.5f} ') 
        
            if (valid_AUC > best_val_auc):
                best_epoch = epoch
                best_val_auc = valid_AUC
                best_source_model = source_model
                torch.save(best_source_model.state_dict(), source_path)

        logger.info('保存的最佳模型轮数:{}'.format(best_epoch))


    best_source_model.load_state_dict(torch.load(source_path, map_location='cpu'), strict=True)
    test_Y, test_P, _, test_Accuracy, test_Precision, test_Recall, test_AUC, test_PRC, test_Sensitivity, test_Specificity, test_F1= test(best_source_model, device, src_test_loader, split="test")
    logger.info( f'source_test ' +
                f'test_AUC: {test_AUC:.5f} ' +
                f'test_PRC: {test_PRC:.5f} ' +
                f'test_Accuracy: {test_Accuracy:.5f} ' +
                f'test_Precision: {test_Precision:.5f} ' +
                f'test_Sensitivity: {test_Sensitivity:.5f} ' +
                f'test_Specificity: {test_Specificity:.5f} ' +
                f'test_F1: {test_F1:.5f} ' +
                f'test_Reacll: {test_Recall:.5f} ')
    
    test_result_savepath = f'result/{args.source_dataset_select}_{args.da_strat}_{args.al_strat}{args.Suffix}'
    if not os.path.exists(test_result_savepath):
            os.makedirs(test_result_savepath)
            
    with open(test_result_savepath + '/'+ args.al_runs + 'test_results.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                # 写入表头
                writer.writerow(["test_AUC", "test_PRC", "test_Accuracy", "test_Precision", "test_Sensitivity", "test_Specificity", "test_F1", "test_Recall"])
                writer.writerow([
                        f"{test_AUC:.5f}", 
                        f"{test_PRC:.5f}", 
                        f"{test_Accuracy:.5f}", 
                        f"{test_Precision:.5f}", 
                        f"{test_Sensitivity:.5f}", 
                        f"{test_Specificity:.5f}", 
                        f"{test_F1:.5f}", 
                        f"{test_Recall:.5f}"
                    ]) 
    if args.task != 'None_DA':
        run_active_adaptation(args, best_source_model, src_dset, device, logger, cfg)

if __name__ == '__main__':
    for time in range(1) :
        main(time)

