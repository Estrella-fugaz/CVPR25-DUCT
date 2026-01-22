import logging
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import copy
import ot

from methods.base import BaseLearner
from models.vit_inc import IncViTCosine
from utils.toolkit import state_dict_to_vector, vector_to_state_dict


class DUCT(BaseLearner):
    remove_keys = ['vit_backbone.head', 'fc.']

    def __init__(self, args):
        super().__init__(args)
        self._network = IncViTCosine(args)

        self.args = args

        self.train_loader = None
        self.task_cl_inc = self.args['increment']

        self.epochs = args['epochs']
        self.lrate = args['lrate']
        self.lrate_decay = args['lrate_decay']
        self.weight_decay = args['weight_decay']
        self.batch_size = args['batch_size']

        if self.args.get('bcb_lr_scale', False):
            self.bcb_lr_scale = args['bcb_lr_scale']
        else:
            self.bcb_lr_scale = 0.01

        self.model_init_state = copy.deepcopy(self._network.state_dict())
        self.model_init_vector = (state_dict_to_vector(self.model_init_state, remove_keys=self.remove_keys)
                                  .to(self._device))
        self.model_merged_vector = torch.zeros_like(self.model_init_vector).cpu()

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        self._network.update_fc(self._total_classes)
        self._network.to(self._device)

        self._network.load_state_dict(self.model_init_state, strict=False)
        self._compute_class_mean(data_manager)

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                 source="train", mode="train")
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                      num_workers=self.num_workers)

        self._train(self.train_loader, self.test_loader, mode='train')
        self._stage1_merge_backbone(data_manager)
        self._train(self.train_loader, self.test_loader, mode='retrain')
        self._stage2_transport_classifier()

    def _train(self, train_loader, test_loader, mode='train'):
        assert mode in ['train', 'retrain']

        base_params = self._network.vit_backbone.parameters()

        base_fc_params = [p for p in self._network.fc.parameters() if p.requires_grad]
        head_scale = 1.

        bcb_lr = self.lrate * self.bcb_lr_scale
        if self.args.get('lr_last', False) and self._cur_task == self.args['total_sessions'] - 1:
            bcb_lr = self.args["lr_last"]
            logging.info('Last session lr: {}'.format(bcb_lr))

        base_params = {'params': base_params, 'lr': bcb_lr,
                       'weight_decay': self.weight_decay}
        base_fc_params = {'params': base_fc_params, 'lr': self.lrate * head_scale,
                          'weight_decay': self.weight_decay}

        if mode == 'retrain':
            network_params = [base_fc_params]
            lrate = self.args['lr_re'] if self.args.get('lr_re', False) else self.lrate
            epochs = self.args['epc_re'] if self.args.get('epc_re', False) else 5
        else:
            network_params = [base_params, base_fc_params]
            lrate = self.lrate
            epochs = self.epochs
            if not self._cur_task:
                epochs = self.args['epc_init'] if self.args.get('epc_init', False) else epochs
                logging.info('Initial epochs: {}'.format(epochs))

        optimizer = optim.SGD(network_params, lr=lrate, momentum=0.9, weight_decay=self.weight_decay)
        # scheduler = optim.lr_scheduler.ConstantLR(optimizer=optimizer)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args['milestones'], gamma=0.1)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        for epoch in range(1, epochs + 1):

            self._network.train()
            losses = 0.
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                logits = self._network(inputs)["logits-l"]

                cur_targets = torch.where(targets - self._known_classes >= 0, targets - self._known_classes, -100)

                if self.args.get('cls_fixed', False) and self.args['cls_fixed']:
                    loss = F.cross_entropy(logits, cur_targets)
                else:
                    loss = F.cross_entropy(logits[:, self._known_classes:], cur_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            if epoch % 5 == 0:
                train_acc = self._compute_accuracy_domain(self._network, train_loader)
                test_acc = self._compute_accuracy_domain(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.3f}, Test_accy {:.3f}'.format(
                    self._cur_task, epoch, epochs, losses / len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}'.format(
                    self._cur_task, epoch, epochs, losses / len(train_loader))
            logging.info(info)

        if mode == 'retrain':
            test_acc = self._compute_accuracy_domain(self._network, self.test_loader)
            logging.info('Test accuracy of model after retraining: {}'.format(test_acc))

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    @torch.no_grad()
    def _stage1_merge_backbone(self, data_manager):

        vector_model_updated = state_dict_to_vector(self._network.state_dict(), remove_keys=self.remove_keys)
        vector_model_updated = vector_model_updated.to(self._device)
        vector_model_inc = (vector_model_updated - self.model_init_vector).detach()

        merge_scalar = 0.99
        if self.args.get('merge_scalar', False):
            merge_scalar = self.args['merge_scalar']

        cl_means_updated = self._cal_means_cur(data_manager)
        self.model_merged_vector += vector_model_inc.cpu().squeeze() * merge_scalar

        if self.args.get('continual_merge', False) and self.args['continual_merge']:
            self.model_init_vector = vector_model_inc.squeeze() * merge_scalar + self.model_init_vector
            self._network.load_state_dict(
                vector_to_state_dict(self.model_init_vector.to(self._device), self._network.state_dict(),
                                     remove_keys=self.remove_keys), strict=False)

        del vector_model_updated, vector_model_inc

        if self.args.get('continual_merge', False) and self.args['continual_merge']:
            pass
        else:
            self._network.load_state_dict(
                vector_to_state_dict(self.model_merged_vector.to(self._device) + self.model_init_vector,
                                     self._network.state_dict(),
                                     remove_keys=self.remove_keys), strict=False)

        test_acc = self._compute_accuracy_domain(self._network, self.test_loader)
        logging.info('Test accuracy of model after backbone merge: {}'.format(test_acc))


    @torch.no_grad()
    def _stage2_transport_classifier(self):
        def _get_task_range(task_id):
            task_start = task_id * self.task_cl_inc
            task_end = (task_id + 1) * self.task_cl_inc
            return task_start, task_end

        if not self._cur_task:
            return

        self._network.eval()

        cur_range_start, cur_range_end = _get_task_range(self._cur_task)
        cur_head_weight = self._network.fc.weight[cur_range_start:cur_range_end]
        cur_cl_means = torch.tensor(self._class_means[cur_range_start:cur_range_end]).to(self._device)

        for old_task_id in range(self._cur_task):
            old_range_start, old_range_end = _get_task_range(old_task_id)
            old_head_weight = self._network.fc.weight[old_range_start:old_range_end]
            old_cl_means = torch.tensor(self._class_means[old_range_start:old_range_end]).to(self._device)

            cost_matrix = torch.cdist(cur_cl_means, old_cl_means, p=2)
            prob_source = torch.ones(self.task_cl_inc).to(self._device) / self.task_cl_inc * 1.0
            prob_target = torch.ones(self.task_cl_inc).to(self._device) / self.task_cl_inc * 1.0
            reg = self.args["ot_reg"] if self.args.get("ot_reg", False) else 0.02
            T_matrix = ot.sinkhorn(prob_source, prob_target, cost_matrix, reg)  # @return: torch.DoubleTensor

            old_head_weight_transported = torch.matmul(T_matrix.T, cur_head_weight.to(torch.double)).to(torch.float32)
            ot_cost = np.multiply(T_matrix.cpu().numpy(), cost_matrix.cpu().numpy())
            ot_cost = ot_cost.sum()
            merge_ratio = self.args['head_merge_ratio'] if self.args.get('head_merge_ratio', False) else 0.5

            old_head_weight_updated = (1 - merge_ratio) * old_head_weight + merge_ratio * old_head_weight_transported
            self._network.fc.weight[old_range_start:old_range_end] = old_head_weight_updated

        test_acc = self._compute_accuracy_domain(self._network, self.test_loader)
        logging.info('Test accuracy of model after classifier transport: {}'.format(test_acc))

    def _cal_means_cur(self, data_manager):
        class_means = []
        for class_id in range(self._known_classes, self._total_classes):
            class_data = data_manager.get_dataset([class_id], source='train', mode='train')
            class_loader = DataLoader(class_data, batch_size=self.batch_size, shuffle=False,
                                      num_workers=self.num_workers)
            vectors, _ = self._extract_vectors(class_loader)
            class_mean = np.mean(vectors, axis=0)
            class_means.append(class_mean)

        class_means = np.array(class_means)

        return class_means
