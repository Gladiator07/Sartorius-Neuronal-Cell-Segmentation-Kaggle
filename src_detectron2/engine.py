import wandb
import numpy as np

import detectron2.data.transforms as T
from detectron2.data import DatasetCatalog, build_detection_train_loader, DatasetMapper
from detectron2.engine import DefaultTrainer
from detectron2.evaluation.evaluator import DatasetEvaluator
from torch.optim.lr_scheduler import OneCycleLR

# local modules
from utils import calculate_score


class MAPIOUEvaluator(DatasetEvaluator):
    """
    Custom evaluator class for calculating competition metric while validation

    Args:
        DatasetEvaluator (class): Base evaluator class from detectron2
    """

    def __init__(self, dataset_name):
        dataset_dicts = DatasetCatalog.get(dataset_name)
        self.annotations_cache = {
            item["image_id"]: item["annotations"] for item in dataset_dicts
        }

    def reset(self):
        self.scores = []

    def process(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            if len(out["instances"]) == 0:
                self.scores.append(0)
            else:
                targ = self.annotations_cache[inp["image_id"]]
                self.scores.append(calculate_score(out, targ))

    def evaluate(self):
        wandb.log({"map_iou": np.mean(self.scores)})
        return {"map_iou": np.mean(self.scores)}


# Sub-classing Trainer class to add custom stuff
class Trainer(DefaultTrainer):
    """
    Custom trainer class to add custom hooks for training

    Args:
        DefaultTrainer (class): `DefaultTrainer` base class from detectron2

    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return MAPIOUEvaluator(dataset_name)

    # @classmethod
    # def build_train_loader(cls, cfg):
    #     min_size = cfg.INPUT.MIN_SIZE_TRAIN
    #     max_size = cfg.INPUT.MAX_SIZE_TRAIN
    #     sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        
    #     return build_detection_train_loader(cfg,
    #     mapper = DatasetMapper(cfg, is_train=True, augmentations=[
    #         T.ResizeShortestEdge(min_size, max_size, sample_style),
    #         # T.RandomFlip(horizontal=True, vertical=False, prob=0.5),
    #         # T.RandomFlip(horizontal=False, vertical=True, prob=0.5)
    #     ]))