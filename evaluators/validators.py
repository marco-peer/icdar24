from pathlib import Path
import logging
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import normalize
import torch

from tqdm import tqdm
import numpy as np

from evaluators.retrieval import Retrieval
from utils.collate import collate_fn_evaluate
from utils.whitening import whiten

def pca(pfs, args):
    pca_dim = args['eval_options']['pca_dim']
    
    pca = None

    if args['eval_options'].get('apply_pca', True):
        if pca_dim != -1 and min(pfs.shape) > 500:
            pca_dim = min(min(pfs.shape), pca_dim)
            print(f'Fitting PCA with shape {pca_dim}')

            pca = PCA(pca_dim, whiten=True)
            pfs_tf = pca.fit_transform(pfs)
            pfs = normalize(pfs_tf, axis=1)

    elif args['eval_options'].get('only_whitening', False):
        logging.info(f'Perform whitening on page descriptors {pfs.shape}')
        pfs = normalize(whiten(pfs, 'pca'), axis=1)

    return pfs, pca

class SIFTEvaluator:
    """If dataset consists of sift patches."""

    def __init__(self, validation_set, page_encoding, args):
        self.validation_set = validation_set
        self.page_encoding = page_encoding
        self.args = args
        self.evaluator = Retrieval()
        
    
    def eval(self, model):
        model.eval()
        # feats, writers, pages = self.inference(model)
        # pfs, writer = self.compute_page_features(feats, writers, pages)
        pfs, writer = self.inference(model)

        descs = np.concatenate(pfs) # self.page_encoding(pfs)
        descs, _ = pca(descs, self.args)

        logger_result, csv_result = self.evaluator.eval(descs, writer)
        
        return logger_result, csv_result
        
    def inference(self, model):
        model.eval()
        loader = torch.utils.data.DataLoader(self.validation_set, num_workers=4, batch_size=self.args['test_batch_size'])

        feat_dim = self.args['model_options']['in_dim'] if self.args['model_options']['in_dim'] != -1 else  model.backbone.embed_dim
        final_dim = feat_dim *  model.nv.num_clusters
        feats = np.zeros((len(self.validation_set), final_dim))
        print(f'pre-init feature array with shape {feats.shape}')
        pages = []
        writers = []
        # feats = []
        idx = 0
        for sample, labels in tqdm(loader, desc='Inference'):
            w,p = labels[0], labels[1]

            writers.append(w.numpy())
            pages.append(p.numpy())
            sample = sample.cuda()

            with torch.no_grad():
                emb = model(sample)
                emb = torch.nn.functional.normalize(emb)
            f = emb.detach().cpu().numpy()
            feats[idx:idx+f.shape[0]] = f
            idx = idx+f.shape[0]
  
        
        writers = np.concatenate(writers)
        pages = np.concatenate(pages)   

        print(writers.shape, pages.shape)
        # _labels = list(zip(writers, pages))
        _labels = np.unique(np.stack([writers, pages]), axis=1)
        # _labels = set(_labels)
        print(_labels.shape)
        page_features = []
        page_writer = []
        page_pages = []
      


        if 'grk50' in self.args['training'].lower():
            logging.info(f'Computing PCA with shape {feats.shape}')
            dim = feats.shape[-1]
            fitting = np.linspace(0, feats.shape[0]-1, max(20000, feats.shape[-1])).astype(np.int32)
            pca = PCA(dim, whiten=True).fit(feats[fitting])
            feats = normalize(pca.transform(feats))

        for i in tqdm(range(_labels.shape[1])):
            idx = np.where((writers == _labels[0, i]) & (pages == _labels[1, i]))
            page_features.append(self.page_encoding([feats[idx]]))
            page_writer.append(_labels[0, i])
            page_pages.append(_labels[1, i])
        

        if 'grk50' in self.args['training'].lower():
            pfs = np.concatenate(page_features)
            logging.info(f'Page descriptors {pfs.shape}')
            logging.info(f'Writers {len(page_writer)}')
            logging.info(f'Pages {len(page_pages)}')

            np.save(Path(self.args['log_dir']) / 'pfs.npy', pfs)
            np.save(Path(self.args['log_dir']) / 'writers.npy', np.stack(page_writer))
            np.save(Path(self.args['log_dir']) / 'pages.npy', np.stack(page_pages))   
        

        return page_features, page_writer


    def compute_page_features(self, _features, writer, pages):
        _labels = list(zip(writer, pages))

        labels_np = np.array(_labels)

        writer = labels_np[:, 0]
        page = labels_np[:, 1]

        labels_np = None
        _labels = set(_labels)

        page_features = []
        page_writer = []

        for w, p in tqdm(_labels, 'Page Features'):
            idx = np.where((writer == w) & (page == p))
            page_features.append(_features[idx])
            page_writer.append(w)

        return page_features, page_writer

class PageEvaluator:
    """If dataset consists of full pages"""

    def __init__(self, validation_set, page_encoding, args, num_samples=None):
        self.validation_set = validation_set
        self.page_encoding = page_encoding
        self.args = args
        self.num_samples = num_samples
        self.evaluator = Retrieval()
    
    @torch.no_grad()
    def eval(self, model):
        model.eval()
        model.backbone.eval()
        dataloader = torch.utils.data.DataLoader(self.validation_set, batch_size=1, shuffle=False, collate_fn=collate_fn_evaluate, num_workers=8, pin_memory=True)
        embs, labels = [], []
        pages = []
        for sample, label in tqdm(dataloader):
            inp = sample.cuda()


            # emb = model.backbone(inp)
            # desc = model.fc(model.forward_features(emb.unsqueeze(-1).unsqueeze(-1).permute(-1, 1, 2, 0)))
            # embs.append(desc.detach().cpu().numpy())
            if inp.shape[0] > 5000: # max batch_size is 2000 
                emb_t = []
                inps = inp.chunk(int(inp.shape[0] // 5000) + 1)
                for inp in inps:
                    emb = model(inp).detach().cpu().numpy()
                    emb_t.append(emb)
                
                embs.append(self.page_encoding(np.concatenate(emb_t)))
                
            else:

                emb = model(inp).detach().cpu().numpy()
                embs.append(self.page_encoding(emb))

            labels.append(label[0][0].item())
            pages.append(label[0][1].item())

        pfs = np.concatenate(embs)

        logging.info(f'Page descriptors {pfs.shape}')
        logging.info(f'Writers {len(labels)}')
        logging.info(f'Pages {len(pages)}')

        np.save(Path(self.args['log_dir']) / 'pfs.npy', pfs)
        np.save(Path(self.args['log_dir']) / 'writers.npy', np.stack(labels))
        np.save(Path(self.args['log_dir']) / 'pages.npy', np.stack(pages))   
        
        pfs, _ = pca(pfs, self.args)
        logger_result, csv_result = self.evaluator.eval(pfs, labels)

        return logger_result, csv_result