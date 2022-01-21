import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Union
from argparse import Namespace

# detectron2 deps
import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import inference_on_dataset
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, build_detection_test_loader

# local modules
from utils import (
    calculate_score,
    BitMasks__init__,
)


def ths_cal_setup(
    args: Namespace,
    data_cfg: dict,
    model_weights_path: Path,
) -> Dict:
    """
    Initial setup for threshold calculation (registering COCO instances, config file merging)

    Args:
        args (Namespace): passed command line arguments to ArgumentParser
        data_cfg (dict): dictionary having all necessary paths
        model_weights_path (Path): model weights path for which threshold is to be calculated

    Returns:
        Dict: configuration
    """
    print(f"[INFO] Setting configuration for threshold calculation ...")
    cfg = get_cfg()
    cfg.INPUT.MASK_FORMAT = "bitmask"

    if args.val_split == "tts_split":
        train_annotations_path = data_cfg["train_tts_sb_annotations"]
        val_annotations_path = data_cfg["val_tts_sb_annotations"]
    elif args.val_split == "5fold_split":
        train_annotations_path = data_cfg["train_5fold_annotations"]
        val_annotations_path = data_cfg["val_5fold_annotations"]
    try:
        print(f"[INFO] Registering COCO instances ...")
        print(f"[INFO] Registering train annotations from {train_annotations_path}")
        print(f"[INFO] Registering val annotations from {val_annotations_path}")

        register_coco_instances(
            "sartorius_train",
            {},
            train_annotations_path,
            data_cfg["root_dir_data"],
        )
        register_coco_instances(
            "sartorius_val",
            {},
            val_annotations_path,
            data_cfg["root_dir_data"],
        )
        print(f"[INFO] sartorius_train, sartorius_val registered successfully")

    except Exception as e:
        print(f"\n{e}")

    cfg.merge_from_file(model_zoo.get_config_file(args.model_name))
    # cfg.merge_from_file(args.config_path, allow_unsafe=True)
    cfg.MODEL.WEIGHTS = model_weights_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    # just for sanity check (#TODO: Remove this after figuring out the pipeline)
    cfg.INPUT.MASK_FORMAT = "bitmask"
    if args.environ == "jarvislabs":
        cfg.DATALOADER.NUM_WORKERS = 7
    else:
        cfg.DATALOADER.NUM_WORKERS = os.cpu_count()  # for colab, kaggle, paperspace

    print(
        f"[INFO] Using {cfg.DATALOADER.NUM_WORKERS} workers for data loading (inference)"
    )
    print(f"[INFO] Loading config file from {args.model_name}")
    print(f"[INFO] Loading model weights from {model_weights_path}")

    return cfg


class MAPIOUCEvaluator(DatasetEvaluator):
    """
    Custom evaluator class for calculating competition metric per class

    Args:
        DatasetEvaluator (class): Base evaluator class from detectron2
    """

    def __init__(self, dataset_name, classes=3):
        dataset_dicts = DatasetCatalog.get(dataset_name)
        self.annotations_cache = {
            item["image_id"]: item["annotations"] for item in dataset_dicts
        }
        self.classes = classes

    def reset(self):
        self.scores = []

    def process(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            if len(out["instances"]) == 0:
                self.scores.append([0] * self.classes)
            else:
                targ = self.annotations_cache[inp["image_id"]]
                class_scores = []
                for c in range(self.classes):
                    targ_c = [x for x in targ if x["category_id"] == c]
                    if targ_c:
                        class_scores.append(calculate_score(out, targ))
                    else:
                        class_scores.append(0)
                self.scores.append(class_scores)

    def evaluate(self):
        return {"map_iou": np.mean(self.scores, axis=0).tolist()}


def calculate_optimal_threshold(
    cfg,
    start_threshold=0.05,
    threshold_step=0.05,
    score_per_pixel=False,
    calculate_optimal_min_pixels=False,  # TODO: integrate min pixel calculator as well
) -> Union[pd.DataFrame, list]:
    """
    Calculate optimal thresholds per class

    Args:
        cfg (dict) : configuration (yaml)
        start_threshold (float, optional): Start threshold. Defaults to 0.05.
        threshold_step (float, optional): Step to take for calculating thresholds (start_threshold to 1). Defaults to 0.05.
        score_per_pixel (bool, optional): Whether to generate confidence scores per pixel. Defaults to False.
        calculate_optimal_min_pixels (bool, optional): Whether to calculate min pixels to include predictions. Defaults to False.calculate_optimal_min_pixels (bool, optional): [description]. Defaults to False.

    Returns:
        Union[pd.DataFrame, list]: threshold_scores_df (scores for different threshold for each class),
                                    optimal_thresholds (list of optimal thresholds per class)
                                    order of cell_type ["shsy5y", "astro", "cort"]
    """
    if score_per_pixel:
        print("\n[INFO] Scoring predictions per pixel ...")
        detectron2.structures.masks.BitMasks.__init__.__code__ = (
            BitMasks__init__.__code__
        )

    ths_scores = []

    val_loader = build_detection_test_loader(
        cfg, "sartorius_val", num_workers=cfg.DATALOADER.NUM_WORKERS
    )

    for th in np.arange(start_threshold, 1, threshold_step):
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(th)
        model = DefaultPredictor(cfg)
        infer_result = inference_on_dataset(
            model.model, val_loader, MAPIOUCEvaluator("sartorius_val")
        )
        print(th, "->", infer_result)
        th_scores = [th]
        th_scores.extend(infer_result["map_iou"])
        ths_scores.append(th_scores)

    ths_scores_df = pd.DataFrame(ths_scores)
    ths_scores_df.columns = ["confidence_threshold", "shsy5y", "astro", "cort"]
    ths_1 = np.round(
        ths_scores_df.iloc[ths_scores_df["shsy5y"].idxmax(), 0], 2
    )  # shsy5y
    ths_2 = np.round(ths_scores_df.iloc[ths_scores_df["astro"].idxmax(), 0], 2)  # astro
    ths_3 = np.round(ths_scores_df.iloc[ths_scores_df["cort"].idxmax(), 0], 2)  # cort

    optimal_thresholds = [ths_1, ths_2, ths_3]
    print("\n" + "-" * 10 + f" OPTIMAL THRESHOLDS -> {optimal_thresholds} " + "-" * 10)

    return ths_scores_df, optimal_thresholds
