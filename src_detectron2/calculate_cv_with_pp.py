import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from argparse import Namespace

# detectron2 deps
import pycocotools.mask as mask_util
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog

# local modules
from utils import precision_at


def calculate_final_cv(
    args: Namespace,
    model_weights_path=None,
    optimal_thresholds: list = [0.5, 0.5, 0.5],
) -> float:
    """
    Calculate final cv score with following post-process:
    - threshold per class
    - removing overlapping predictions
    - excluding masks with smaller area (by taking top 1 percentile as cut-off)

    Args:
        args (Namespace): passed command line arguments to ArgumentParser
        optimal_thresholds (list, optional): calculated optimal thresholds per class. Defaults to [0.5, 0.5, 0.5].

    Returns:
        float: final cv score
    """

    MIN_PIXELS = [50, 110, 60]
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.model_name))
    cfg.INPUT_MASK_FORMAT = "bitmask"
    cfg.MODEL.WEIGHTS = model_weights_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    if args.environ == "jarvislabs":
        cfg.DATALOADER.NUM_WORKERS = 7
    else:
        cfg.DATALOADER.NUM_WORKERS = os.cpu_count()  # for colab, kaggle, paperspace

    print(
        f"[INFO] Using {cfg.DATALOADER.NUM_WORKERS} workers for data loading (inference)"
    )
    predictor = DefaultPredictor(cfg)
    val_ds = DatasetCatalog.get("sartorius_val")
    scores = []
    p_bar = tqdm(val_ds, total=len(val_ds))
    p_bar.set_description("Calculating final CV with post-process: ")
    for data in p_bar:
        file_name = data["file_name"]
        im = cv2.imread(str(file_name))
        pred = predictor(im)
        # HACK: ugly hack to prevent errors while debugging
        # take the highest occuring cell type
        try:
            pred_class = torch.mode(pred["instances"].pred_classes)[0]
        except:
            print(f"[INFO] Pred class couldn't be determined, setting it to 0")
            pred_class = 0
        # take the scores which are greater than threshold for a particular class
        take = pred["instances"].scores >= optimal_thresholds[pred_class]
        pred_masks = pred["instances"].pred_masks[take]
        pred_masks = pred_masks.cpu().numpy()
        if args.score_per_pixel:
            used = np.zeros(im.shape[:2])
        else:
            used = np.zeros(im.shape[:2], dtype=int)
        fin_pred_masks = []
        for mask in pred_masks:
            mask = mask * (1 - used)
            # TODO : exclude predictions with smaller area (calculated based on statistics)
            # (maybe in future versions or just set it to constant by calculating top 1 percentile cut-off as shown here)
            # https://www.kaggle.com/julian3833/sartorius-classifier-mask-r-cnn-lb-0-28?scriptVersionId=78194908&cellId=10
            if mask.sum() >= MIN_PIXELS[pred_class]:
                used += mask
                fin_pred_masks.append(mask)
        enc_preds = [
            mask_util.encode(np.asarray(p, order="F", dtype=np.uint8))
            for p in fin_pred_masks
        ]
        enc_targs = list(map(lambda x: x["segmentation"], data["annotations"]))
        ious = mask_util.iou(enc_preds, enc_targs, [0] * len(enc_targs))

        prec = []
        for t in np.arange(0.5, 1.0, 0.05):
            tp, fp, fn = precision_at(t, ious)
            p = tp / (tp + fp + fn)
            prec.append(p)

        scores.append(np.mean(prec))
    final_cv = np.mean(scores)

    return final_cv
