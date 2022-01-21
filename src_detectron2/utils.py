import os
import cv2
import torch
import random
import wandb
import shutil
import gspread
import argparse
import numpy as np
import pycocotools.mask as mask_util

from typing import Tuple, Union
from pathlib import Path
from datetime import date
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultPredictor

# From https://www.kaggle.com/stainsby/fast-tested-rle
def rle_decode(mask_rle: str, shape: Tuple = (520, 704)) -> np.array:
    """
    Decode RLE into numpy array image

    Args:
        mask_rle (str): string formatted run-length
        shape (Tuple, optional): (height, width) of array to return. Defaults to (520, 704).

    Returns:
        np.array: reshaped image (1 - mask, 0 - background)
    """

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def rle_encode(img: np.array) -> str:
    """
    Encodes numpy array image into RLE

    Args:
        img (np.array): Image in form of numpy array

    Returns:
        str: RLE string
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def precision_at(threshold, iou) -> float:
    """
    Calculate true positive, false positive and false negative at a given threshold

    Args:
        threshold (float): Threshold for calculating Iou
        iou (np.array): A numpy array having Ious for all masks

    Returns:
        float: tp, fp, fn
    """
    matches = iou > threshold  # boolean array (True if iou > threshold else False)
    # all of the below statements are boolean mask
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = (
        np.sum(true_positives),
        np.sum(false_positives),
        np.sum(false_negatives),
    )
    return tp, fp, fn


# A single prediction from detectron2 contains:
# https://www.notion.so/atharvaingle/Sartorius-Cell-Instance-Segmentation-0199366f6605439c99bdbfa83aaa7796#41bbfafe7b874160b51b7cf06dd5cb8a
def calculate_score(pred: dict, targ: dict) -> float:
    """
    Calculates competition metric

    Args:
        pred (dict): predictions from detectron2 `DefaultPredictor`
        targ (dict): target dictionary having annotations

    Returns:
        score [float]: final competition metric
    """
    # pred_masks: a boolean tensor (True - mask present, False - background)
    # encodes the complete image (torch.Size([# of instances detected, 520, 704]))
    pred_masks = pred["instances"].pred_masks.cpu().numpy()
    enc_preds = [
        mask_util.encode(np.asarray(p, dtype=np.uint8, order="F")) for p in pred_masks
    ]
    enc_targs = list(map(lambda x: x["segmentation"], targ))

    # compute iou between masks
    ious = mask_util.iou(enc_preds, enc_targs, [0] * len(enc_targs))
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, ious)
        p = tp / (tp + fp + fn)
        prec.append(p)

    return np.mean(prec)


def copy_file_to_wandb_dir(file_path, file_name):
    """
    Copy a file to current wandb running directory (for logging purpose to wandb)

    Args:
        file_path (str): path of file to copy
        file_name (str): file name to save in wandb run dir
    """
    shutil.copyfile(file_path, f"{wandb.run.dir}/{file_name}")


def asHours(seconds):
    """
    Returns seconds to human-readable formatted string

    Args:
        seconds (float): total seconds

    Returns:
        str: total seconds converted to human-readable formatted string
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:.0f}h:{m:.0f}m:{s:.0f}s"


def seed_everything(seed):
    """
    Seed all of the randomness

    Args:
        seed (int): seed number
    """
    print(f"[INFO] Global Seed set to: {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def str2bool(v):
    """
    Function to enable argparse to use bool values as flags

    Args:
        v (bool): True or False

    Raises:
        argparse.ArgumentTypeError: Boolean value expected.

    Returns:
        bool: True or False
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "True", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "False", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def remove_files(files_to_keep: list, data_dir: Path):
    """
    Remove files other than the provided list of files

    Args:
        files_to_keep (list): a list of filenames to keep
        data_dir (Path): directory containing `files_to_keep`
    """
    # iterate over files in directory
    for filename in os.listdir(f"{data_dir}"):
        if filename in files_to_keep:
            f = os.path.join(f"{data_dir}", filename)
            # checking if it is a file
            if os.path.isfile(f):
                print(f"Saving file {f}")
        else:
            print(f"Deleting {filename}")
            os.remove(f"{data_dir}/{filename}")


# Per pixel scoring
def BitMasks__init__(self, tensor: Union[torch.Tensor, np.ndarray]):
    """
    Replaces the detectron2 default function to get per pixel score
    Args:
        tensor: bool Tensor of N,H,W, representing N instances in the image.
    """
    device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
    tensor = torch.as_tensor(
        tensor, dtype=torch.float32, device=device
    )  # Original code: tensor = torch.as_tensor(tensor, dtype=torch.bool, device=device)
    assert tensor.dim() == 3, tensor.size()
    self.image_size = tensor.shape[1:]
    self.tensor = tensor


def write_to_google_sheet(credential_file_path: Path, metadata_file_obj):
    """
    Write important information to google sheet

    Args:
        credential_file_path (Path): credentials.json file path
        metadata_file_obj (dict): metdata.json file object
    """
    print(f"[INFO] Writing info to Google sheet ...")
    gc = gspread.service_account(filename=credential_file_path)
    sh = gc.open_by_key("1FOldQ3U40-S6AxmG8E7OOhpDFu27HWGYuLXMx_PU-lU")
    worksheet = sh.sheet1
    # dd/mm/YY
    today_date = date.today().strftime("%d-%m-%Y")
    model = metadata_file_obj["model_name"]
    experiment = metadata_file_obj["experiment_name"]
    run_url = metadata_file_obj["wandb_run_link"]
    configuration = metadata_file_obj["configuration"]
    fold = metadata_file_obj["fold"]
    cv_raw = metadata_file_obj["cv"]
    cv_with_pp = metadata_file_obj["cv_with_pp"]
    public_lb = ""
    comment = metadata_file_obj["comment"]
    environ = metadata_file_obj["environ"]

    if environ == "colab":
        environ = "colab pro"
    worksheet.append_row(
        [
            today_date,
            model,
            experiment,
            configuration,
            fold,
            cv_raw,
            cv_with_pp,
            public_lb,
            comment,
            environ,
            run_url,
        ]
    )

    print("Done...")


def log_val_predictions_wandb(cfg: dict, n_images: int = 20) -> None:
    """
    Function to log ground truth and predictions from the trained model for further analyzing purpose

    Args:
        cfg (dict): configuration
        n_images (int, optional): Total images to log. Defaults to 20.
    """

    wandb_table = wandb.Table(columns=["id", "ground_truth", "predicted"])

    cfg.MODEL.WEIGHTS = os.path.join(
        cfg.OUTPUT_DIR, "model_best.pth"
    )  # path to the model just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    dataset_dicts = DatasetCatalog.get("sartorius_val")
    outs = []

    # TO-DO - selecting the images randomly, may change this to get same images
    for d in random.sample(dataset_dicts, n_images):
        im = cv2.imread(d["file_name"])
        image_id = d["image_id"]

        # outputs from model
        # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        outputs = predictor(im)

        # TO-DO - log confidence scores or mean if required
        # print(outputs["instances"].scores.to("cpu"))  [maybe log scores in future]

        v = Visualizer(
            im[:, :, ::-1],
            metadata=MetadataCatalog.get("sartorius_train"),
            instance_mode=ColorMode.IMAGE_BW
            # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        # draw masks on predicted images
        out_pred = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # ground truth masks
        visualizer = Visualizer(
            im[:, :, ::-1], metadata=MetadataCatalog.get("sartorius_train")
        )
        out_target = visualizer.draw_dataset_dict(d)

        # log stuff to wandb table
        ground_truth_imgs = wandb.Image(out_target.get_image()[:, :, ::-1])
        predicted_imgs = wandb.Image(out_pred.get_image()[:, :, ::-1])

        wandb_table.add_data(image_id, ground_truth_imgs, predicted_imgs)

    wandb.log({"Val predictions": wandb_table})
