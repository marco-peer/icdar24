import logging, os, argparse

import torch
import torch.backends.cudnn as cudnn
from dataloading.dataset import DocumentDataset

from mae.trainer import Trainer
from utils.augmentations import get_page_transforms, get_patch_transforms
from utils.utils import GPU, save_json, seed_everything, load_config, getLogger

from mae.hog_openmim import MaskedAutoencoderViT

def get_mae(args, finetune_args={}):
    m = MaskedAutoencoderViT(img_size=args['img_size'], patch_size=args['model_options'].get('patch_size', 8), embed_dim=args['model_options'].get('embed_dim', 368),hog_pool=args['model_options'].get('hog_pool', 4), hog_bins=9, 
                            decoder_depth=args['model_options'].get('decoder_depth', 8), depth=args['model_options'].get('depth', 8), in_chans=3,
                            global_pool=finetune_args.get('model_options', {}).get('global_pool', False), 
                            norm_pix_loss=not args['model_options'].get('use_hog', True), target_in_chans=3 if args['train_options']['mask_type'] == 'rgb' else 1
                        )

    return m
    
def prepare_logging(args):
    os.path.join(args['log_dir'], args['super_fancy_new_name'])
    Logger = getLogger(args["logger"])
    logger = Logger(os.path.join(args['log_dir'], args['super_fancy_new_name']), args=args)
    logger.log_options(args)
    return logger

def main(args):
    logger = prepare_logging(args)
    logger.update_config(args)

    model = get_mae(args).cuda()

    if args['checkpoint']:
        checkpoint = torch.load(args['checkpoint'])
        model.load_state_dict(checkpoint['model_state_dict']) 

    training_page_transforms = get_page_transforms(args)
    training_patch_transforms = get_patch_transforms(args)

    train_ds = DocumentDataset(args['training'])
    val_ds = DocumentDataset(args['validation'])
    test_ds = DocumentDataset(args['testing'])


    training = train_ds.SamplerWithBinary(page_transforms=training_page_transforms, patch_size=args['img_size'], num_keypoints=args['train_options']['num_keypoints'],
                                          mask_type=args['train_options']['mask_type'], binarization=args['train_options']['binarization'], num_samples=args['train_options']['num_samples'])    
    validation = val_ds.SelectLabels(label_names='writer').EvalSIFTSampler(patch_size=args['img_size'], num_samples=args['val_options']['num_samples'])
    testing = test_ds.SelectLabels(label_names='writer').EvalSIFTSampler(patch_size=args['img_size'], num_samples=args['eval_options']['num_samples'])
    
    trainer = Trainer(args, training, validation, logger, training_patch_transforms)
    if not args['only_test']:
        trainer.train(model)

    logger_result, csv_result = trainer.validate(model, dataset=testing)
    logger.log_value('Test-mAP', logger_result['map'])
    logger.log_value('Top-1', logger_result['top1'])
    save_json(csv_result, os.path.join(logger.log_dir, 'test_results.json'))
    
    # testing
    logger.finish()

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s ')
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='config/maskfeat.yml')
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

    run_name = config['train_options']['mask_type'] + '_' + config['train_options']['binarization'] + '_MASK' + str(config['train_options']['mask_ratio']) + '_' + str(config['model_options']['embed_dim']) + '_HOG' + str(config['model_options']['use_hog'])
    config['super_fancy_new_name'] = config.get('super_fancy_new_name', run_name)

    GPU.set(args.gpuid, 400)
    cudnn.benchmark = True
    
    seed_everything(args.seed)
    
    main(config)