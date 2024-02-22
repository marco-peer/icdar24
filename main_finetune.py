from functools import partial
import logging, os, argparse, math, random, re, glob
import pickle as pk
from pathlib import Path
from cv2 import SIFT
from matplotlib.dviread import Page
from tqdm import tqdm

import numpy as np
import torch
from torch import optim
import torch.backends.cudnn as cudnn

from torchvision import transforms

from sklearn.decomposition import PCA

from finetune.trainer import Finetuner
from finetune.netvlad import Wrapper
from page_encodings import get_encoder
from utils.augmentations import Dilation, Erosion

from utils.utils import GPU, save_json, seed_everything, load_config, getLogger, save_model, cosine_scheduler, prepare_logging, load_yaml
from utils.collate import collate_fn_evaluate
from evaluators.validators import SIFTEvaluator, PageEvaluator
from dataloading.dataset import DocumentDataset

import torch.multiprocessing
from main_maskfeat import get_mae

import sklearn.cluster

torch.set_num_threads(8)


def train_val_split(dataset, prop = 0.9):
    authors = list(set(dataset.labels['writer']))
    random.shuffle(authors)

    train_len = math.floor(len(authors) * prop)
    train_authors = authors[:train_len]
    val_authors = authors[train_len:]

    print(f'{len(train_authors)} authors for training - {len(val_authors)} authors for validation')

    train_idxs = []
    val_idxs = []

    for i in tqdm(range(len(dataset)), desc='Splitting dataset'):
        w = dataset.labels['writer'][i]
        if w in train_authors:
            train_idxs.append(i)
        if w in val_authors:
            val_idxs.append(i)

    return train_idxs, val_idxs

def load_model(checkpoint_path, args):
    model_zoo = load_yaml('config/model_zoo.yml')
    h = model_zoo.get(args['model_zoo']['experiment'], {}).get(args['model_zoo']['model'], None)
    checkpoint_path = h or checkpoint_path

    logging.info(f'Loading model from {checkpoint_path}')
    yml = Path(checkpoint_path).parent / 'config.yaml'
    model_config = load_yaml(yml)
    model = get_mae(model_config, finetune_args=args)

    if args.get('load_checkpoint', True):
        checkpoint = torch.load(checkpoint_path)
        msg = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if args.get('model_options', {}).get('global_pool', False):
            assert set(msg.missing_keys) == {'fc_norm.weight', 'fc_norm.bias', 'gmp.lamb'}
    else:
        logging.info('Not loading checkpoint - training from scratch')
        
    model.decoder_blocks = torch.nn.Identity()
    model.decoder_pred = torch.nn.Identity()
    model.eval()
    return model

def get_finetune_model(backbone, args):
    if args['finetune_options'].get('freeze_backbone', False):
        logging.info('Freezing backbone')
        for m in backbone.parameters():
            m.requires_grad = False

    ## setup final model
    m = Wrapper(backbone, args)
    m.eval()
    m = m.cuda()
    return m

def cluster(descs, num_centers):
    if num_centers == 1:
        return np.mean(descs, axis=0, keepdims=True)
    kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=num_centers, compute_labels=False, init='k-means++', n_init='auto',
                                             batch_size=10000)

    kmeans.fit(descs)
    
    c = kmeans.cluster_centers_
    logging.info(f'Clustering done - centers: {c.shape}')
    return c

def get_pca_components(descs, out_dim):
    pca = PCA(n_components=out_dim, whiten=True)
    descs = pca.fit_transform(descs)
    return descs, pca.components_

@torch.no_grad()
def init_netvlad(train_ds, model, args):
    descs = []
    loader =  torch.utils.data.DataLoader(train_ds, batch_size=2000, num_workers=16)
    model.eval()

    for sample, _ in tqdm(loader, desc='Inference for initialization'):
        sample = sample.cuda()
        embs = model.forward_features(sample)

        descs.append(embs.detach().cpu().numpy())

    descs = np.concatenate(descs, axis=0)

    if in_dim := args.get('model_options', {}).get('in_dim', -1):
        if in_dim != -1:
            logging.info(f'Fitting PCA for netvlad from {descs.shape[1]} to {in_dim}')
            descs, components = get_pca_components(descs, in_dim)
            model.embed_fc.weight.data = torch.tensor(components).cuda()
            # model.embed_fc.bias.data = torch.zeros(in_dim).cuda()

    logging.info(f'Clustering netvlad centroids from {descs.shape[0]} descriptors')
    vlad_centers = torch.from_numpy(cluster(descs, args['netvlad']['num_clusters'])).cuda()
    model.nv._init_params(clusters=vlad_centers)
    return model


def get_tfs(args):
    tfs = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    val_tfs = list(tfs)

    tfs = [tfs[0]]
    tfs.extend([transforms.RandomApply(
            [Erosion()],
            p=0.3
        ),
        transforms.RandomApply(
            [Dilation()],
            p=0.3
        )
        ])
    
    if 'grk50' in args['training'].lower():
        tfs.append(
            transforms.RandomAffine(10, translate=(0.1,0.1), scale=(0.9, 1.1), shear=None, interpolation=transforms.InterpolationMode.NEAREST, fill=1)
        )
    if 'icdar2017'.lower() in args['training'] and 'color' in args['training'].lower():
        n = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        tfs.append(transforms.Grayscale(num_output_channels=3))
        tfs.append(n)
        val_tfs = [transforms.ToTensor(), transforms.Grayscale(num_output_channels=3), n]
        return tfs, val_tfs
    
    tfs.append(val_tfs[1])

    return tfs, val_tfs

def main(args):

    logger = prepare_logging(args)
    logger.update_config(args)
    args['log_dir'] = logger.log_dir
    
    #############################################
    ######## Get model and transforms

    if args['model_zoo']['experiment'] == 'resnet56':
        from models.resnets import resnet56
        backbone = resnet56()
        backbone.num_features = backbone.embed_dim = 64
        logging.info("Using resnet56 as backbone")
    elif args['model_zoo']['experiment'] == 'resnet20':
        from models.resnets import resnet20
        backbone = resnet20()
        backbone.num_features = backbone.embed_dim = 64
        logging.info("Using resnet20 as backbone")
    else:
        backbone = load_model(args['finetune_options'].get('checkpoint', None), args)
        logging.info(f'Using {args["model_zoo"]["model"]} as backbone')

    model = get_finetune_model(backbone, config)

    if args['checkpoint']:
        checkpoint = torch.load(args['checkpoint'])
        model.load_state_dict(checkpoint['model_state_dict'])

    # del model.backbone.blocks[-2:]

    # for i, block in enumerate(model.backbone.blocks[:4]):
    #     logging.info(f'Freezing block {i}')
    #     for p in block.parameters():
    #         p.requires_grad = False

    if getattr(model.backbone, 'blocks', None):
        logging.info(f'Using {len(model.backbone.blocks)} blocks as backbone')

    train_tf, val_tf = [transforms.Compose(i) for i in get_tfs(args)]
    print(train_tf, val_tf)
    #############################################
    ######## Setup datasets

    # if dataset consists of SIFT patches already extracted (unsupervised)

    d = DocumentDataset(args['training'])
    if args.get('validation', None) is None:
        train_idxs, val_idxs = train_val_split(d, prop=args.get('train_authors_prop', 0.9))
        train_dataset = d.SelectLabels(label_names=args['train_label'])
        val_dataset = d.SelectLabels(label_names=['writer', 'page'])
        train_len = len(train_idxs)
        train = torch.utils.data.Subset(train_dataset.TransformImages(transform=train_tf), train_idxs)
        val = torch.utils.data.Subset(val_dataset.TransformImages(transform=val_tf), val_idxs)
    else:
        train_dataset = d.SelectLabels(label_names=args['train_label'])
        train = train_dataset.TransformImages(transform=train_tf)
        train_len = len(train_dataset)
        train_idxs = range(train_len)
        val_dataset = DocumentDataset(args['validation'])
        if 'patch' not in args['validation']:
            # full pages
            if 'color' in args['validation']:
                val_dataset = val = val_dataset.SelectLabels(label_names=['writer']).EvalSIFTColorSampler(patch_size=args.get('img_size', 32), num_samples=args['val_options']['num_samples'])
            else:
                val_dataset = val =  val_dataset.SelectLabels(label_names=['writer']).EvalSIFTSampler(patch_size=args.get('img_size', 32), num_samples=args['val_options']['num_samples'])
        else:
            # patches
            val_dataset = val = val_dataset.SelectLabels(label_names=['writer', 'page']).TransformImages(transform=val_tf)


    # test set
    test_ds = DocumentDataset(args['testing'])
    test = test_ds.SelectLabels(label_names=['writer', 'page'])

    if 'hisfrag' in args['testing'].lower() or 'color' in args['testing'].lower():
        # full pages in color
        test = test.EvalSIFTColorSampler(patch_size=args.get('img_size', 32), num_samples=-1)
    elif 'patches' in args['testing'].lower(): 
        test = test_ds.SelectLabels(label_names=['writer', 'page']).TransformImages(transform=val_tf)
    else:
        # full pages binarized
        test = test.EvalSIFTSampler(patch_size=args.get('img_size', 32), num_samples=-1)



    #################################################
    #################################################
    ######## init model
    
    if args.get('init_netvlad', True) and not args['only_test']:
        length = min(train_len, 200000)
        n = int(train_len / length)

        init_dataset = torch.utils.data.Subset(train_dataset.TransformImages(transform=val_tf), train_idxs[::n])
        model = init_netvlad(init_dataset, model, args)

    #################################################
    #################################################
    ########## Training
    
    if 'patch' in args.get('validation', args['training']):
        validator = SIFTEvaluator(val, get_encoder(args), args)
    else:
        validator = PageEvaluator(val, get_encoder(args), args)

    if not args['only_test']:
        trainer = Finetuner(args, train, validator, logger)
        model = trainer.train(model)

    #################################################
    #################################################

    ## Evaluating on Test set
        
    if 'patch' in args.get('testing', ''):
        test_evaluator = SIFTEvaluator(test, get_encoder(args), args)
    else:
        test_evaluator = PageEvaluator(test, get_encoder(args), args, num_samples=args['eval_options'].get('num_samples', -1))

    # test_evaluator = PageEvaluator(test, get_encoder(args), args, num_samples=args['eval_options'].get('num_samples', -1))

    logger_result, csv_result = test_evaluator.eval(model)
    logger.log_value('Test-mAP', logger_result['map'])
    logger.log_value('Top-1', logger_result['top1'])
    save_json(csv_result, os.path.join(logger.log_dir, 'test_results.json'))

    logger.finish()


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s ')
    parser = argparse.ArgumentParser()
    
    # torch.multiprocessing.set_start_method('spawn')
    
    parser.add_argument('--config', default='config/finetune_maskfeat_icdar2017.yml')
    parser.add_argument('--only_test', default=False, action='store_true',
                        help='only test')
    parser.add_argument('--checkpoint', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--gpuid', default='2', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--seed', default=2174, type=int,
                        help='seed')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='debug - turn wandb off')

    args = parser.parse_args()
        
    if args.debug:
        logging.info('wandb disabled')
        os.environ['WANDB_MODE'] = "offline"

    config = load_config(args)[0]

    run_name = config['model_zoo']['experiment'] + '_' + str(config['model_zoo']['model']) + "_" + config['train_label']
    config['super_fancy_new_name'] = config.get('super_fancy_new_name', run_name)


    GPU.set(args.gpuid, 400)
    cudnn.benchmark = True
    
    seed_everything(args.seed)
    main(config)