import logging
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import torch, torchvision
from tqdm import tqdm
from pytorch_metric_learning import samplers, miners, losses, distances

from evaluators.retrieval import Retrieval

from page_encodings import get_encoder
from utils.collate import collate_fn_mae, collate_fn_evaluate
from utils.utils import cosine_scheduler, save_model, get_params_groups

class Finetuner:

    def __init__(self, args, train_ds, validator, logger, aug = None):
        self.args = args
        
        self.training = train_ds
        self.validator = validator

        self.length_before_new_iter = self.args['train_options'].get('length_before_new_iter', len(train_ds))
        self.scheduler = cosine_scheduler(args['optimizer_options']['base_lr'], args['optimizer_options']['final_lr'], self.args['train_options']['epochs'], int(self.length_before_new_iter / self.args['train_options']['batch_size']), warmup_epochs=args['optimizer_options']['warmup_epochs'], start_warmup_value=args['optimizer_options']['start_lr'])
        self.wd_scheduler = cosine_scheduler(args['optimizer_options']['base_wd'], args['optimizer_options']['final_wd'], self.args['train_options']['epochs'], int(self.length_before_new_iter  / self.args['train_options']['batch_size']), warmup_epochs=args['optimizer_options']['warmup_epochs'], start_warmup_value=args['optimizer_options']['start_wd'])
        scale = self.args['train_options']['batch_size'] / self.args['optimizer_options'].get('base_batch_size', self.args['train_options']['batch_size'])
        self.scheduler *= scale

        self.logger = logger

        if self.args['train_options'].get('loss', 'triplet') == 'triplet':
            from utils.triplet_loss import TripletLoss
            logging.info('Using Triplet loss with margin 0.1')
            self.loss = TripletLoss(margin=self.args['train_options'].get('margin', 0.1))
            self.miner = None
        elif self.args['train_options'].get('loss', 'triplet') == 'triplet_semihard':
            logging.info('Using triplet loss with semihard miner')
            self.loss = losses.TripletMarginLoss(margin=0.1)
            self.miner = miners.TripletMarginMiner(margin=0.1, type_of_triplets='semihard')

        else:
            logging.info('Using ms loss with miner with specified settings')
            self.loss =  losses.MultiSimilarityLoss(alpha=self.args['train_options'].get('loss_a', 2), beta=self.args['train_options'].get('loss_b', 40), base=self.args['train_options'].get('loss_base', 0.5))
            self.miner = miners.MultiSimilarityMiner(epsilon=self.args['train_options'].get('mining_margin', 0.1))

        self.aug = aug
        self.page_encoding = get_encoder(args)
        self.retrieval = Retrieval()

        self.scaler = torch.cuda.amp.GradScaler() if self.args['train_options'].get('mixed_precision', False) else None
        
    def _setup_optimizer(self, parameters):
        if self.args['optimizer_options']['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(parameters, lr=self.args['optimizer_options']['base_lr'], momentum=0.9) 
        elif self.args['optimizer_options']['optimizer'] == 'adamw':
            self.optimizer = torch.optim.AdamW(parameters, lr=self.args['optimizer_options']['base_lr'], betas=(0.9, 0.999))
        else:
            self.optimizer = torch.optim.Adam(parameters, lr=self.args['optimizer_options']['base_lr'], weight_decay=self.args['optimizer_options'].get('wd', 0))

    def _update_learning_rate(self, it):
        for i, param_group in enumerate(self.optimizer.param_groups):
            # set up lr
            param_group['weight_decay'] = self.args['optimizer_options']['base_wd']

            if it > (len(self.scheduler) - 1):
                param_group['lr'] = self.scheduler[-1]
            else:
                param_group["lr"] = self.scheduler[it]

            # modify NetVLAD lr
            if param_group['name'] == 'netvlad':
                if it == 0:
                    logging.info(f"NetVLAD learning rate is multiplied by factor {self.args['optimizer_options'].get('netvlad_lr_factor', 1)}")
                param_group['lr'] *= self.args['optimizer_options'].get('netvlad_lr_factor', 1)

            # modify pca lr
            if param_group['name'] == 'pca':
                if it == 0:
                    logging.info(f"PCA learning rate is multiplied by factor {self.args['optimizer_options'].get('pca_lr_factor', 1)}")
                param_group['lr'] *= self.args['optimizer_options'].get('pca_lr_factor', 1)

            if param_group['name'] == 'gmp':
                if it == 0:
                    logging.info(f"GMP learning rate is multiplied by factor {self.args['optimizer_options'].get('gmp_lr_factor', 1)}")
                param_group['lr'] *= self.args['optimizer_options'].get('pca_lr_factor', 1)
                
            # set up weight decay and learning rate decay for ViT
            if 'backbone' in param_group['name']:
                if it == 0:
                    logging.info(f"Weight decay active for backbone")

                if param_group.get('decay', False):
                    param_group["lr"] *= param_group['decay']

                    if it == 0:
                        logging.info(f"{param_group['decay']} for {param_group['name']}")



    def _train_one_epoch(self, model, epoch):
        model.train()
        model = model.cuda()

        if getattr(self.training, 'indices', None):
            labels = np.array(self.training.dataset.labels[self.args['train_label']])[self.training.indices]
        else:
            labels = np.array(self.training.labels[self.args['train_label']])

        collate = None if 'patch' in self.args['training'] else collate_fn_evaluate

        # set up the sampling stuff
        sampler = samplers.MPerClassSampler(labels, self.args['train_options']['sampler_m'], batch_size=self.args['train_options']['batch_size'], length_before_new_iter=self.length_before_new_iter)#, self.args['train_options']['length_before_new_iter']) 
        train_triplet_loader = torch.utils.data.DataLoader(self.training, sampler=sampler, batch_size=self.args['train_options']['batch_size'], drop_last=True, num_workers=32, collate_fn=collate, pin_memory=True)

        pbar = tqdm(train_triplet_loader)
        pbar.set_description('Epoch {} Training'.format(epoch))
        iters = len(train_triplet_loader)
        self.logger.log_value('Epoch', epoch, commit=False)

        for i, (samples, label) in enumerate(pbar):
            it = iters * epoch + i
            self._update_learning_rate(it)
        
            samples = samples.cuda()
            samples.requires_grad=True
            label = label.cuda()
            
            embs = model(samples)
            if self.miner is not None:
                loss = self.loss(embs, label, self.miner(embs, label))
            else:
                loss = self.loss(embs, label, embs, label)

            self.logger.log_value(f'loss', loss.item())
            self.logger.log_value(f'lr', self.optimizer.param_groups[0]['lr'])
            self.logger.log_value(f'wd', self.wd_scheduler[min(it, len(self.wd_scheduler) - 1)])

            # compute gradient and update weights
            self.optimizer.zero_grad()
            loss.backward()
            if self.args['clip_gradients']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args['clip_gradients'])
            self.optimizer.step()
            # model.nv.centroids.data = torch.nn.functional.normalize(model.nv.centroids.data, dim=1)

        torch.cuda.empty_cache()
        return model
    
    def get_parameters(self, model):
        
        parameters = [{"params": model.nv.parameters(), "lr": self.args['optimizer_options']['start_lr'] * self.args['optimizer_options']['netvlad_lr_factor'], 'name' : 'netvlad',  'weight_decay' : self.args['optimizer_options']['base_wd']},
                      {"params": model.embed_fc.parameters(), "lr": self.args['optimizer_options']['start_lr'] * self.args['optimizer_options']['pca_lr_factor'], 'name' : 'pca',  'weight_decay' : self.args['optimizer_options']['base_wd']}]

        if self.args['optimizer_options'].get('layer_decay', False):
            decay = self.args['optimizer_options'].get('layer_decay')
            for decay_power, i in enumerate(range(len(model.backbone.blocks)-1, -1, -1)):
                d = {"params": model.backbone.blocks[i].parameters(), 
                     "lr": self.args['optimizer_options']['start_lr'], 
                     'name' : f'backbone_{i}', 
                     'decay' : decay ** (decay_power +1),
                     'weight_decay' : self.args['optimizer_options']['base_wd']}
                parameters.append(d)

            # patch embed, pos emb, ...
            for p in ['patch_embed', 'pos_embed', 'norm', 'fc_norm', 'gmp']:
                layer =  getattr(model.backbone, p, None)
                if layer is None:
                    continue
                param = layer.parameters() if isinstance(layer, torch.nn.Module) else layer
                parameters.append(
                    {"params": param, 
                    "lr": self.args['optimizer_options']['start_lr'], 
                    'name' : p, 
                    'weight_decay' : self.args['optimizer_options']['base_wd'],
                    'decay' : 1}
                )
                logging.info(f'Setting parameters for {p}')
        else:
            parameters.append({"params": model.backbone.parameters(), "lr": self.args['optimizer_options']['start_lr'], 'name' : 'backbone'})
        
        return parameters
    
    def train(self, model):

        parameters = self.get_parameters(model)
        self._setup_optimizer(parameters)

        res, _ = self.validator.eval(model)
        best_mAP = res['map']
        logging.info(f'Epoch -1 - Val-mAP: {best_mAP}')

        self.logger.log_value('Val-mAP', best_mAP)
        best_epoch = -1

        for epoch in range(self.args['train_options']['epochs']):
            model.train()
            model = self._train_one_epoch(model, epoch)

            res, _ = self.validator.eval(model)
            val_map = res['map']
            logging.info(f'Epoch {epoch} - Val-mAP: {val_map}')
            self.logger.log_value('Val-mAP', val_map)
            self.logger.log_value('Val-Top1', res['top1'])
            self
            if val_map > best_mAP:
                best_epoch = epoch
                best_mAP = val_map
                save_model(model, self.optimizer, epoch, os.path.join(self.logger.log_dir, 'model.pt'))

            if epoch - best_epoch > self.args['train_options'].get('early_stopping', self.args['train_options']['epochs'] + 1):
                # if no early stopping is defined, it is deactivated
                break
        
        if self.args['train_options'].get('use_best_model', True):
            checkpoint = torch.load(os.path.join(self.logger.log_dir, 'model.pt'))
            print(f'''Loading model from Epoch {checkpoint['epoch']}''')
            model.load_state_dict(checkpoint['model_state_dict'])    
            model.eval()
        else:
            logging.info("Using model from last epoch")
            save_model(model, self.optimizer, epoch, os.path.join(self.logger.log_dir, 'model.pt'))

        return model
    