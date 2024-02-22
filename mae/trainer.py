import logging
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import torch, torchvision
from torchvision import transforms
from tqdm import tqdm
from pytorch_metric_learning import samplers

from evaluators.retrieval import Retrieval

from page_encodings import get_encoder
from utils.collate import collate_fn_mae, collate_fn_self_supervised, collate_fn_evaluate
from utils.dino_losses import DINOLoss, KoLeoLoss
from utils.utils import cosine_scheduler, save_model, get_params_groups

class Trainer:

    def __init__(self, args, train_ds, validate_ds, logger, aug):
        self.args = args
        
        self.training = train_ds
        self.validation = validate_ds

        self.scheduler = cosine_scheduler(args['optimizer_options']['base_lr'], args['optimizer_options']['final_lr'], self.args['train_options']['epochs'], int(len(train_ds) / self.args['train_options']['batch_size']), warmup_epochs=args['optimizer_options']['warmup_epochs'], start_warmup_value=args['optimizer_options']['start_lr'])
        self.wd_scheduler = cosine_scheduler(args['optimizer_options']['base_wd'], args['optimizer_options']['final_wd'], self.args['train_options']['epochs'], int(len(train_ds) / self.args['train_options']['batch_size']), warmup_epochs=args['optimizer_options']['warmup_epochs'], start_warmup_value=args['optimizer_options']['start_wd'])
        scale = self.args['train_options']['batch_size'] / self.args['optimizer_options'].get('base_batch_size', self.args['train_options']['batch_size'])
        self.scheduler *= scale

        self.logger = logger
        self.mask_ratio = args['train_options']['mask_ratio']

        self.aug = aug
        self.page_encoding = get_encoder(args)
        self.retrieval = Retrieval()

        self.scaler = torch.cuda.amp.GradScaler() if self.args['train_options'].get('mixed_precision', False) else None
        
    def _setup_optimizer(self, model):
        param_groups = get_params_groups(model)

        if self.args['optimizer_options']['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(param_groups, lr=self.args['optimizer_options']['base_lr'], momentum=0.9) 
        elif self.args['optimizer_options']['optimizer'] == 'adamw':
            self.optimizer = torch.optim.AdamW(param_groups, lr=self.args['optimizer_options']['base_lr'], weight_decay=self.args['optimizer_options']['base_wd']) #weight_decay=self.args['optimizer_options'].get('wd', 0))
        else:
            self.optimizer = torch.optim.Adam(param_groups, lr=self.args['optimizer_options']['base_lr'], weight_decay=self.args['optimizer_options'].get('wd', 0))

    def _update_learning_rate(self, it):
        for i, param_group in enumerate(self.optimizer.param_groups):
            if i == 0:
                if it > (len(self.wd_scheduler) - 1):
                    param_group['weight_decay'] = self.wd_scheduler[-1]
                else:
                    param_group["weight_decay"] = self.wd_scheduler[it]
                
            if it > (len(self.scheduler) - 1):
                param_group['lr'] = self.scheduler[-1]
            else:
                param_group["lr"] = self.scheduler[it]

    def train(self, model):
        self._setup_optimizer(model)


        res, _ = 0,0 
        best_mAP = 0
        logging.info(f'Epoch -1 - Val-mAP: {best_mAP}')

        self.logger.log_value('Val-mAP', best_mAP)
        best_epoch = -1

        for epoch in range(self.args['train_options']['epochs']):
            model.train()
            self._train_one_epoch(model, epoch)

            res, _ = self.validate(model)
            val_map = res['map']
            logging.info(f'Epoch {epoch} - Val-mAP: {val_map}')
            self.logger.log_value('Val-mAP', val_map)

            if val_map > best_mAP:
                best_epoch = epoch
                best_mAP = val_map
                save_model(model, self.optimizer, epoch, os.path.join(self.logger.log_dir, 'best_model.pt'))
            if epoch % self.args['save_every_n'] == 0:
                save_model(model, self.optimizer, epoch, os.path.join(self.logger.log_dir, f'model_{epoch}.pt'))

            if epoch - best_epoch > self.args['train_options'].get('early_stopping', self.args['train_options']['epochs'] + 1):
                # if no early stopping is defined, it is deactivated
                break

        save_model(model, self.optimizer, epoch, os.path.join(self.logger.log_dir, f'model_final.pt'))

        checkpoint = torch.load(os.path.join(self.logger.log_dir, 'best_model.pt'))
        print(f'''Loading model from Epoch {checkpoint['epoch']}''')
        model.load_state_dict(checkpoint['model_state_dict'])    
        model.eval() 
        return model
    
    def validate(self, model, dataset=None):
        model.eval()
        dataloader = torch.utils.data.DataLoader(dataset or self.validation, batch_size=1, shuffle=False, collate_fn=collate_fn_evaluate, num_workers=8, pin_memory=True)
        embs, labels = [], []
        bs = 800
        for (sample, label) in tqdm(dataloader):
            inp = sample.cuda()

            if inp.shape[0] > bs:
                emb_t = []
                inps = inp.chunk(int(inp.shape[0] // bs) + 1)
                for inp in inps:
                    emb = torch.nn.functional.normalize(model.extract_features(inp).detach().cpu()).numpy()
                    emb_t.append(emb)
                
                embs.append(self.page_encoding(np.concatenate(emb_t)))
                
            else:
                emb = torch.nn.functional.normalize(model.extract_features(inp).detach().cpu()).numpy()
                embs.append(self.page_encoding(emb))

            labels.append(label[0].item())
            
        pfs = np.concatenate(embs)
        pca_dim = self.args['eval_options']['pca_dim']

        if pca_dim != -1 and min(pfs.shape) > 600:
            pca_dim = min(min(pfs.shape), pca_dim)
            print(f'Fitting PCA with shape {pca_dim}')

            pca = PCA(pca_dim, whiten=True)
            pfs_tf = pca.fit_transform(pfs)
            pfs = normalize(pfs_tf, axis=1)
        
        logger_result, csv_result = self.retrieval.eval(pfs, labels)
        return logger_result, csv_result
        

    def _train_one_epoch(self, model, epoch):
        dataloader = torch.utils.data.DataLoader(self.training, batch_size=self.args['train_options']['batch_size'], shuffle=True, collate_fn=collate_fn_mae, num_workers=16, pin_memory=True)
        iters_per_epoch = int(len(self.training) / self.args['train_options']['batch_size'])
        self.logger.log_value('Epoch', epoch)
            
        pbar = tqdm(dataloader)
        pbar.set_description('Epoch {} Training'.format(epoch))

        losses = 0
        bs = 0
        for idx, (sample, binarized) in enumerate(pbar):
            it = iters_per_epoch * epoch + idx
            self._update_learning_rate(it)
            inp = self.aug(sample.cuda())

            if self.scaler is None:
                loss, pred, mask = model(inp, hog_target_imgs = binarized.cuda(), mask_ratio=self.mask_ratio)
                self.optimizer.zero_grad()
                loss.backward()

                if self.args.get('clip_gradients', None) and self.args.get('clip_gradients', None) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args['clip_gradients'])

                self.optimizer.step()
            else:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    loss, pred, mask = model(inp, hog_target_imgs = binarized.cuda(), mask_ratio=self.mask_ratio)
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    
                    if self.args.get('clip_gradients', None) and self.args.get('clip_gradients', None) > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args['clip_gradients'])

                    self.scaler.step(self.optimizer)

                    # Updates the scale for next iteration.
                    self.scaler.update()

            losses += loss.detach()
            bs += inp.shape[0]

            if idx % int((iters_per_epoch * 0.05)) == 0:

                self.logger.log_value('loss', losses/bs)
                self.logger.log_value('learning_rate', self.scheduler[it])
                self.logger.log_value('wd', self.wd_scheduler[it])
                self.logger.log_value('batch_size', inp.shape[0])

            # visualize images of the first epoch
            # if idx == 0:
            #     images = inp.cpu()
            #     binarized = binarized.cpu()

            #     images = torch.cat([images[::4], binarized[::4].repeat(1,3,1,1)], dim=0)
            #     tile = lambda x: torchvision.transforms.ToPILImage()(torchvision.utils.make_grid(images.cpu(), nrow=images.shape[0] // 2))
            #     self.logger.log_image(tile(images), epoch, 'train_imgs')

        return model
