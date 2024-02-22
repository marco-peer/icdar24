# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import cv2  # type: ignore

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import argparse
import json
import os
from typing import Any, Dict, List

from tqdm import tqdm
import glob

from utils.utils import GPU

parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help=(
        "Path to the directory where masks will be output. Output will be either a folder "
        "of PNGs per image or a single json with COCO-style masks."
    ),
)

parser.add_argument(
    "--model-type",
    type=str,
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
    default='vit_h'
)

parser.add_argument(
    "--checkpoint",
    type=str,
    help="The path to the SAM checkpoint to use for mask generation.",
    default='/data/mpeer/sam_vit_h_4b8939.pth'
)

parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

parser.add_argument(
    "--convert-to-rle",
    action="store_true",
    help=(
        "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
        "Requires pycocotools."
    ),
)

amg_settings = parser.add_argument_group("AMG Settings")

amg_settings.add_argument(
    "--points-per-side",
    type=int,
    default=8,
    help="Generate masks by sampling a grid over the image with this many points to a side.",
)

amg_settings.add_argument(
    "--points-per-batch",
    type=int,
    default=8,
    help="How many input points to process simultaneously in one batch.",
)

amg_settings.add_argument(
    "--pred-iou-thresh",
    type=float,
    default=None,
    help="Exclude masks with a predicted score from the model that is lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-thresh",
    type=float,
    default=0.8,
    help="Exclude masks with a stability score lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-offset",
    type=float,
    default=None,
    help="Larger values perturb the mask more when measuring stability score.",
)

amg_settings.add_argument(
    "--box-nms-thresh",
    type=float,
    default=0.01,
    help="The overlap threshold for excluding a duplicate mask.",
)

amg_settings.add_argument(
    "--crop-n-layers",
    type=int,
    default=None,
    help=(
        "If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."
    ),
)

amg_settings.add_argument(
    "--crop-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding duplicate masks across different crops.",
)

amg_settings.add_argument(
    "--crop-overlap-ratio",
    type=int,
    default=None,
    help="Larger numbers mean image crops will overlap more.",
)

amg_settings.add_argument(
    "--crop-n-points-downscale-factor",
    type=int,
    default=None,
    help="The number of points-per-side in each layer of crop is reduced by this factor.",
)

amg_settings.add_argument(
    "--min-mask-region-area",
    type=int,
    default=None,
    help=(
        "Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."
    ),
)

amg_settings.add_argument(
    "--gpuid",
    type=int,
    default=0
)

import numpy as np

def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    # header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    # metadata = [header]

    mask = masks[0]["segmentation"]
    if len(masks) > 1:                          # if we have multiple masks, we try to merge them
        r = masks[0]['area'] / masks[1]['area'] 
        if r < 1.15 and r > 0.85:
            mask = mask + masks[1]["segmentation"]
            mask = np.clip(mask, 0, 1)
    
    img = cv2.imread(masks[0]['imgpath'])[250:-250, 250:-250]
    masked = cv2.bitwise_and(img, img, mask=mask.astype(np.uint8)[250:-250, 250:-250])
    masked = masked[:, ~np.all(cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY) == 0, axis=0)]  # discard black pixels
    masked = masked[~np.all(cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY) == 0, axis=1), :]  # discard black pixels

    cv2.imwrite(path, masked)


def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs


def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
    amg_kwargs = get_amg_kwargs(args)
    generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)
    print("Loading model done!")

    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        targets = [t for t in glob.glob(os.path.join(args.input, '**/*'), recursive=True) if not os.path.isdir(t)]

        # targets = [
        #     f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
        # ]
        # targets = [os.path.join(args.input, f) for f in targets]

    os.makedirs(args.output, exist_ok=True)

    def invert(mask):
        m = mask['segmentation']
        a, b = m.shape
        if np.sum(m[0, :]) > 0.8 * a \
            and  np.sum(m[:, 0]) > 0.8 * b \
            and  np.sum(m[-1, :]) > 0.8 * a \
            and  np.sum(m[:, -1]) > 0.8 * b \
            and np.sum(m) > 10:
            m = 1 - m
        mask['area'] = np.sum(m)
        return mask

    for t in tqdm(sorted(targets)):

        tar = os.path.join(args.output, Path(t).parent.name)
        save_base = os.path.join(tar, f'{Path(t).stem}.png')

        if os.path.exists(save_base):
            continue

        print(f"Processing '{t}'...")
        image = cv2.imread(t)
        if image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        blur = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),(5,5),0)
        image = cv2.Canny(blur, 150, 200)

        # _,image = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        try:
            masks = generator.generate(image)
        except:
            continue
        
        masks.sort(key=lambda x: x['area'], reverse=True)

        new_masks = []
        num_pix = image.shape[0] * image.shape[1]
        

        for i in masks:
            i['imgpath'] = t
            if i['area'] > num_pix * 0.66:  # we found the mask with the handwriting
                new_masks.append(invert(i))
                break
            if i['area'] > num_pix * 0.1:
                new_masks.append(invert(i)) # we found as mask with probably only a small amount of handwriting

        masks = new_masks

        if not masks:
            continue



        if not os.path.exists(tar):
            os.mkdir(tar)

        try:
            write_masks_to_folder(masks, save_base)
        except:
            continue

        # if output_mode == "binary_mask":
        #     os.makedirs(save_base, exist_ok=False)
        #     write_masks_to_folder(masks, save_base)
        # else:
        #     save_file = save_base + ".json"
        #     with open(save_file, "w") as f:
        #         json.dump(masks, f)
    print("Done!")


if __name__ == "__main__":
    args = parser.parse_args()
    GPU.set(args.gpuid)
    main(args)