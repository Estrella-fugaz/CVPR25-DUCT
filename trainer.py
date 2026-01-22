import os
import os.path
import sys
import logging
import copy
import time
import torch
import numpy as np
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters


def train(args):
    seed_list = copy.deepcopy(args['seed'])
    device = copy.deepcopy(args['device'])

    for seed in seed_list:
        args['seed'] = seed
        args['device'] = device
        _train(args)

    myseed = 42069  # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True  # 仅允许 cuDNN 采用 deterministic convolution algorithms [same input leads same output]
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)


def _train(args):
    logfilename = 'logs/{}_{}_{}_{}_{}_{}_{}_'.format(args['prefix'], args['seed'], args['model_name'], args['net_type'],
                                                args['dataset'], args['init_cls'], args['increment'])+ time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    os.makedirs(logfilename)
    print(logfilename)
    args.update({"logfilename": logfilename})

    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(args['dataset'], args['shuffle'], args['seed'], args['init_cls'], args['increment'], args)
    args['class_order'] = data_manager._class_order  # [0, 1, 2, ..., 2069] := 6个任务, 每个任务345类
    model = factory.get_model(args['model_name'], args)

    acc_curve, nme_curve = {'top1': []}, {'top1': []}
    acc_matrix = []
    acc_matrix_nme = []

    for task in range(data_manager.nb_tasks):
        logging.info('All params: {}'.format(count_parameters(model._network)))
        logging.info('Trainable params: {}'.format(count_parameters(model._network, True)))
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        eval_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
        cur_acc_curve = [cnn_accy["grouped"][key] for key in eval_keys]
        # logging.info('Accuracy curve: {}'.format(cur_acc_curve))
        acc_matrix.append(cur_acc_curve)

        model_name = args['model_name'].upper() + '_' + args['net_type'].upper()

        if nme_accy is not None:
            logging.info('{}: {}'.format(model_name, cnn_accy['grouped']))
            logging.info('NME: {}'.format(nme_accy['grouped']))
            cur_acc_nme_curve = [nme_accy['grouped'][key] for key in eval_keys]
            acc_matrix_nme.append(cur_acc_nme_curve)

            acc_curve['top1'].append(cnn_accy['top1'])
            nme_curve['top1'].append(nme_accy['top1'])
            logging.info('{} top1 curve: {}'.format(model_name, acc_curve['top1']))
            logging.info('NME top1 curve: {}'.format(nme_curve['top1']))
            acc_avg_nme = np.mean(nme_curve['top1'])
            logging.info('Average NME top1: {}'.format(acc_avg_nme))
        else:
            logging.info('{}: {}'.format(model_name, cnn_accy['grouped']))
            acc_curve['top1'].append(cnn_accy['top1'])
            logging.info('{} top1 curve: {}'.format(model_name, acc_curve['top1']))

        acc_avg = np.mean(acc_curve['top1'])
        logging.info('Average top1: {}'.format(acc_avg))

        if args.get('save_model', False) and args['save_model']:
            torch.save(model, os.path.join(logfilename, "task_{}.pth".format(int(task))), pickle_protocol=4)

        if args['model_name'] == 'joint' or args['model_name'] == 'linear':
            break

    task_num = args["total_sessions"]
    last_task_id = task_num - 1

    if len(acc_matrix) > 0:
        np_acc_table = np.zeros([task_num, task_num])
        logging.info('Accuracy Matrix:')
        for idx_x, line in enumerate(acc_matrix):
            idx_y = len(line)
            np_acc_table[idx_x, :idx_y] = np.array(line)
            logging.info(np_acc_table[idx_x])
        np_acc_table = np_acc_table.T
        forgetting = np.mean((np.max(np_acc_table, axis=1) - np_acc_table[:, last_task_id])[:last_task_id])

        logging.info('Forgetting (after last task): {}'.format(forgetting))

    if len(acc_matrix_nme) > 0:
        np_acc_table_nme = np.zeros([task_num, task_num])
        logging.info('NME Accuracy Matrix:')
        for idx_x, line in enumerate(acc_matrix_nme):
            idx_y = len(line)
            np_acc_table_nme[idx_x, :idx_y] = np.array(line)
            logging.info(np_acc_table_nme[idx_x])
        np_acc_table_nme = np_acc_table_nme.T
        forgetting_nme = np.mean((np.max(np_acc_table_nme, axis=1) - np_acc_table_nme[:, last_task_id])[:last_task_id])

        logging.info('NME Forgetting (after last task): {}'.format(forgetting_nme))


def _set_device(args):
    device_type = args['device']
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(device))

        gpus.append(device)

    args['device'] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))
