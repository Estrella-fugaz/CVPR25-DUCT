import torch
from torch import nn
from models.vit import _create_vision_transformer
import copy
import logging
import re
import timm
from models.linears import SimpleLinear, CosineLinear, SplitCosineLinear, SimpleContinualLinear


def get_vit_backbone(args):
    model_kwargs = dict(patch_size=16, embed_dim=args["embd_dim"], depth=12, num_heads=12)

    if "backbone" in args:
        vit_cfg_pattern = 'vit_(\w+)_patch(\d+)_\d+'
        match = re.search(vit_cfg_pattern, args["backbone"])
        size_model, size_patch = match.group(1), int(match.group(2))
        model_kwargs.update({'patch_size': size_patch,
                             'num_heads': {'tiny': 3, 'small': 6, 'base': 12, 'large': 16}[size_model]})
        logging.info('You are now using `{}` as backbone.'.format(args["backbone"]))
        try:
            return _create_vision_transformer(args["backbone"], pretrained=True, **model_kwargs)
        except:
            raise ValueError('Unknown backbone: {}.'.format(args["backbone"]))
    else:
        logging.info('You are now using `vit_base_patch16_224` as backbone.')
        return _create_vision_transformer('vit_base_patch16_224', pretrained=True)


class IncViT(nn.Module):
    """
    Enable to learn incremental classifier heads.
    """

    def __init__(self, args: dict) -> None:
        super(IncViT, self).__init__()

        self.vit_backbone = get_vit_backbone(args)

        self.fc = None

        if args["dataset"] == "domainnet":
            self.class_num = 345
            self.fc = nn.Linear(args["embd_dim"], self.class_num, bias=True)
            # self.classifier_pool = nn.ModuleList([
            #     nn.Linear(args["embd_dim"], self.class_num, bias=True)
            #     for i in range(args["total_sessions"])
            # ])
        elif args["dataset"] == "officehome":
            self.class_num = 65
            self.fc = nn.Linear(args["embd_dim"], self.class_num, bias=True)
            # self.classifier_pool = nn.ModuleList([
            #     nn.Linear(args["embd_dim"], self.class_num, bias=True)
            #     for i in range(args["total_sessions"])
            # ])
        elif args["dataset"] == "pacs":
            self.class_num = 7
            self.fc = nn.Linear(args["embd_dim"], self.class_num, bias=True)
        elif args["dataset"] == "cddb":
            self.class_num = 2
            self.fc = nn.Linear(args["embd_dim"], self.class_num, bias=True)
        elif "core50" in args["dataset"]:
            self.class_num = 50
            self.fc = nn.Linear(args["embd_dim"], self.class_num, bias=True)
        else:
            raise ValueError('Unknown datasets: {}.'.format(args["dataset"]))

        self.numtask = 0

        self._device = args["device"][0]

    @property
    def feature_dim(self):
        return self.vit_backbone.out_dim

    def extract_vector(self, x):
        return self.vit_backbone(x)["features"]

    def forward(self, x):
        """
        :return:
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits  ### shape: (N, known_classes + new_classes)
        }
        """
        x = self.vit_backbone(x)
        out = {
            'logits': self.fc(x["features"])
        }
        # out = self.fc(x["features"])
        out.update(x)

        return out

    def interface(self, image, selection):
        # todo: implement this
        pass

    def update_fc(self, nb_classes):
        """
        extension of FC layer -- add new nodes beneath the original FC layer
        NB: supposed to adopt `Xavier initialization` for new class nodes (w.r.t. LwF's paper)
        """
        self.numtask += 1
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim) -> nn.Module:
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

    def un_freeze(self):
        for param in self.parameters():
            param.requires_grad = True
        self.train()

    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            pkl_name = "{}_{}_{}_B{}_Inc{}".format(
                args["dataset"],
                args["seed"],
                args["convnet_type"],
                0,
                args["init_cls"],
            )
            checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        self.vit_backbone.load_state_dict(model_infos['convnet'])
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc


class IncViTCA(IncViT):
    def __init__(self, args, fc_with_ln=False):
        super().__init__(args)
        self.old_fc = None
        self.fc = None
        self.fc_with_ln = fc_with_ln

    def update_fc(self, nb_classes, freeze_old=True):
        if self.fc is None:
            self.fc = self.generate_fc(self.feature_dim, nb_classes)
        else:
            self.fc.update(nb_classes, freeze_old=freeze_old)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleContinualLinear(in_dim, out_dim)

        return fc

    def forward(self, x, bcb_no_grad=False, fc_only=False):
        """
        @param fc_only: whether to process original image input or the feature representations
        """
        if fc_only:
            fc_out = self.fc(x)
            if self.old_fc is not None:
                old_fc_logits = self.old_fc(x)['logits']
                fc_out['old_logits'] = old_fc_logits
            return fc_out
        if bcb_no_grad:
            with torch.no_grad():
                x = self.vit_backbone(x)
        else:
            x = self.vit_backbone(x)
        out = self.fc(x['features'])
        out.update(x)

        return out


class IncViTCosine(IncViT):
    """
    Utilize a cosine classifier head instead of a linear one.
    """

    def __init__(self, args: dict, nb_proxy=1) -> None:
        super().__init__(args)
        self.nb_proxy = nb_proxy
        self.fc = None
        model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
        # if "backbone" in args.keys():
        #     self.vit_backbone = _create_vision_transformer(args["backbone"], pretrained=True, **model_kwargs)
        # else:
        #     self.vit_backbone = _create_vision_transformer('vit_base_patch16_224', pretrained=True, **model_kwargs)
        self.logit_norm = args.get("logit_norm", False)
        self.logit_ensemble = args.get("logit_ensemble", False)

        self.args = args

    def forward(self, x) -> torch.Tensor:
        """
        :return:
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits  ### shape: (N, known_classes + new_classes)
        }
        """
        x = self.vit_backbone(x)
        logit_output_ori = self.fc(x["features"])
        logit_output = logit_output_ori
        if self.logit_norm:
            logit_split = torch.split(logit_output_ori, self.class_num, dim=1)
            logit_softmax = [torch.nn.functional.softmax(logit, dim=1) for logit in logit_split]
            if self.logit_ensemble:
                logit_output = torch.sum(torch.stack(logit_softmax), dim=0)
            else:
                logit_output = torch.cat(logit_softmax, dim=1)
        out = {
            'logits-l': logit_output_ori,
            'logits': logit_output
        }
        # out = self.fc(x["features"])
        out.update(x)

        return out

    def update_fc_split(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            if not hasattr(fc, "fc1"):
                fc.fc1.weight.data = self.fc.weight.data
                fc.sigma.data = self.fc.sigma.data
            else:
                # concatenate the embeddings of the original PTM and adapted PTM
                prev_out_features1 = self.fc.fc1.out_features
                fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
                fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
                fc.sigma.data = self.fc.sigma.data

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.fc is None:
            fc = CosineLinear(in_dim, out_dim, self.nb_proxy, to_reduce=True)
        else:
            prev_out_features = self.fc.out_features // self.nb_proxy
            # prev_out_features = self.fc.out_features
            fc = SplitCosineLinear(
                in_dim, prev_out_features, out_dim - prev_out_features, self.nb_proxy
            )

        return fc

    def update_fc(self, nb_classes):
        if self.fc is None:
            self.fc = CosineLinear(self.feature_dim, nb_classes, self.nb_proxy, to_reduce=False)
        else:
            old_weight = copy.deepcopy(self.fc.weight.data)
            fc = CosineLinear(self.feature_dim, nb_classes, self.nb_proxy, to_reduce=False)
            fc.weight.data[:old_weight.shape[0]] = old_weight
            del self.fc
            self.fc = fc


class IncViTPrompt(nn.Module):
    """
    Specifically for L2P and DualPrompt.
    """

    def __init__(self, args):
        super(IncViTPrompt, self).__init__()
        self.backbone = get_vit_backbone(args)
        if args["get_original_backbone"]:
            self.original_backbone = self.get_original_backbone(args)
        else:
            self.original_backbone = None

    def get_original_backbone(self, args):
        return timm.create_model(
            args["backbone_petl"],
            pretrained=True,
            num_classes=args["nb_classes"],
            drop_rate=args["drop"],
            drop_path_rate=args["drop_path"],
            drop_block_rate=None,
        ).eval()

    def forward(self, x, task_id=-1, train=False):
        with torch.no_grad():
            if self.original_backbone is not None:
                cls_features = self.original_backbone(x)['pre_logits']
            else:
                cls_features = None

        x = self.backbone(x, task_id=task_id, cls_features=cls_features, train=train)
        return x


class IncViTCODAPrompt(nn.Module):
    def __init__(self, args):
        super(IncViTCODAPrompt, self).__init__()
        from models.prompts import CodaPrompt
        self.args = args
        self.backbone = get_vit_backbone(args)
        self.fc = nn.Linear(768, args["nb_classes"])
        self.prompt = CodaPrompt(768, args["total_sessions"], args["prompt_param"])

    # pen: get penultimate features
    def forward(self, x, pen=False, train=False):
        if self.prompt is not None:
            with torch.no_grad():
                q, _ = self.backbone(x)
                q = q[:, 0, :]
            out, prompt_loss = self.backbone(x, prompt=self.prompt, q=q, train=train)
            out = out[:, 0, :]
        else:
            out, _ = self.backbone(x)
            out = out[:, 0, :]
        out = {'features': out.view(out.size(0), -1)}
        if not pen:
            out.update({'logits': self.fc(out['features'])})
        if self.prompt is not None and train:
            return out, prompt_loss
        else:
            return out


class IncViTAdaptive(nn.Module):
    def __init__(self, args):
        super(IncViTAdaptive, self).__init__()
        self.TaskAgnosticExtractor, _ = get_vit_backbone(args)  # Generalized blocks
        self.TaskAgnosticExtractor.train()
        self.AdaptiveExtractors = nn.ModuleList()  # Specialized Blocks
        self.out_dim = None
        self.fc = None
        self.aux_fc = None
        self.task_sizes = []
        self.args = args

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.AdaptiveExtractors)

    def extract_vector(self, x):
        base_feature_map = self.TaskAgnosticExtractor(x)
        features = [extractor(base_feature_map) for extractor in self.AdaptiveExtractors]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        base_feature_map = self.TaskAgnosticExtractor(x)
        features = [extractor(base_feature_map) for extractor in self.AdaptiveExtractors]
        features = torch.cat(features, 1)
        out = {'logits': self.fc(features)}  # {logits: self.fc(features)}

        aux_logits = self.aux_fc(features[:, -self.out_dim:])

        out.update({"aux_logits": aux_logits, "features": features})
        out.update({"base_features": base_feature_map})
        return out

        '''
        {
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        '''

    def update_fc(self, nb_classes):
        _, _new_extractor = get_vit_backbone(self.args)
        if len(self.AdaptiveExtractors) == 0:
            self.AdaptiveExtractors.append(_new_extractor)
        else:
            self.AdaptiveExtractors.append(_new_extractor)
            self.AdaptiveExtractors[-1].load_state_dict(self.AdaptiveExtractors[-2].state_dict())

        if self.out_dim is None:
            # logging.info(self.AdaptiveExtractors[-1])
            self.out_dim = self.AdaptiveExtractors[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, :self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.aux_fc = self.generate_fc(self.out_dim, new_task_size + 1)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = (torch.norm(weights[-increment:, :], p=2, dim=1))
        oldnorm = (torch.norm(weights[:-increment, :], p=2, dim=1))
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print('alignweights,gamma=', gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            pkl_name = "{}_{}_{}_B{}_Inc{}".format(
                args["dataset"],
                args["seed"],
                args["net_type"],
                0,
                args["init_cls"],
            )
            checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        checkpoint_name = checkpoint_name.replace("memo_", "")
        model_infos = torch.load(checkpoint_name)
        model_dict = model_infos['backbone']
        assert len(self.AdaptiveExtractors) == 1

        base_state_dict = self.TaskAgnosticExtractor.state_dict()
        adap_state_dict = self.AdaptiveExtractors[0].state_dict()

        pretrained_base_dict = {
            k: v
            for k, v in model_dict.items()
            if k in base_state_dict
        }

        pretrained_adap_dict = {
            k: v
            for k, v in model_dict.items()
            if k in adap_state_dict
        }

        base_state_dict.update(pretrained_base_dict)
        adap_state_dict.update(pretrained_adap_dict)

        self.TaskAgnosticExtractor.load_state_dict(base_state_dict)
        self.AdaptiveExtractors[0].load_state_dict(adap_state_dict)
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc


class IncViTEaseNet(IncViT):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.inc = args["increment"]
        self.init_cls = args["init_cls"]
        self._cur_task = -1
        self.out_dim = self.vit_backbone.out_dim
        self.fc = None
        self.use_init_ptm = args["use_init_ptm"]
        self.alpha = args["alpha"]
        self.beta = args["beta"]

    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
            # print(name)

    @property
    def feature_dim(self):
        if self.use_init_ptm:
            return self.out_dim * (self._cur_task + 2)
        else:
            return self.out_dim * (self._cur_task + 1)

    # (proxy_fc = cls * dim)
    def update_fc(self, nb_classes):
        self._cur_task += 1

        if self._cur_task == 0:
            self.proxy_fc = self.generate_fc(self.out_dim, self.init_cls).to(self._device)
        else:
            self.proxy_fc = self.generate_fc(self.out_dim, self.inc).to(self._device)

        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        fc.reset_parameters_to_zero()

        if self.fc is not None:
            old_nb_classes = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            fc.weight.data[: old_nb_classes, : -self.out_dim] = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.vit_backbone(x)

    def forward(self, x, test=False):
        if test == False:
            x = self.vit_backbone.forward(x, False)
            out = self.proxy_fc(x)
        else:
            x = self.vit_backbone.forward(x, True, use_init_ptm=self.use_init_ptm)
            if self.args["moni_adam"] or (not self.args["use_reweight"]):
                out = self.fc(x)
            else:
                out = self.fc.forward_reweight(x, cur_task=self._cur_task, alpha=self.alpha, init_cls=self.init_cls,
                                               inc=self.inc, use_init_ptm=self.use_init_ptm, beta=self.beta)

        out = {'logits': out}
        out.update({"features": x})
        return out

    def show_trainable_params(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.numel())


class IncViTBasic(IncViT):
    def __init__(self, args):
        super().__init__(args)
        # for RanPAC
        self.W_rand = None
        self.RP_dim = None

        self.fc = None

    def update_fc(self, nb_classes, next_period_initialization=None):
        if self.RP_dim is not None:
            feature_dim = self.RP_dim
        else:
            feature_dim = self.feature_dim
        fc = self.generate_fc(feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if next_period_initialization is not None:
                weight = torch.cat([weight, next_period_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.vit_backbone(x)["features"]

    def forward(self, x):
        x = self.vit_backbone(x)["features"]
        if self.W_rand is not None:
            x = torch.nn.functional.relu(x @ self.W_rand)
        out = {
            'logits': self.fc(x)
        }
        out.update({
            'features': x,
        })
        return out


class IncViTCosineMultBranch(IncViT):
    def __init__(self, args):
        super().__init__(args)
        self.fc = None

        print('Reconstructing self.backbone with dual branches ...')

        del self.vit_backbone

        self.backbone = torch.nn.Identity()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbones = nn.ModuleList()
        self.args = args

        self.model_type = 'vit'

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self._feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self._feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def forward(self, x):
        features = [backbone(x)["features"] for backbone in self.backbones]
        features = torch.cat(features, 1)

        out = {'logits': self.fc(features)}
        out.update({"features": features})

        return out

    def construct_dual_branch_network(self, tuned_model: IncViTBasic) -> None:
        self.backbones.append(get_vit_backbone(self.args))
        self.backbones.append(tuned_model.vit_backbone)

        self._feature_dim = self.backbones[0].out_dim * len(self.backbones)
        self.fc = self.generate_fc(self._feature_dim, self.args['init_cls'])
