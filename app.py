import os
import streamlit as st
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cellpose import models, io, plot
from pathlib import Path
from tqdm import tqdm

import pycocotools.mask as mask_util


def rle_decode(mask_rle, shape=(520, 704)):
    """
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def rle_encode(img):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def plot_truth_pred_masks(oof: pd.DataFrame, idx):

    orig_rle = oof["annotation"][idx]
    pred_rle = oof["predicted"][idx]
    orig_mask = rle_decode(orig_rle)
    pred_mask = rle_decode(pred_rle)
    print(orig_mask.shape), print(pred_mask.shape)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.set_title("Original mask")
    ax1.imshow(orig_mask, cmap="inferno")
    ax2.set_title("Predicted mask")
    ax2.imshow(pred_mask, cmap="inferno")


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


def calculate_cv_score(oof_df):
    scores = []
    for idx, row in oof_df.iterrows():
        enc_preds = [
            mask_util.encode(
                np.asarray(
                    rle_decode(oof_df["predicted"][idx]), order="F", dtype=np.uint8
                )
            )
        ]
        enc_targs = [
            mask_util.encode(
                np.asarray(
                    rle_decode(oof_df["annotation"][idx]), order="F", dtype=np.uint8
                )
            )
        ]
        ious = mask_util.iou(enc_preds, enc_targs, [0] * len(enc_targs))

        prec = []
        for t in np.arange(0.5, 1.0, 0.05):
            tp, fp, fn = precision_at(t, ious)
            p = tp / (tp + fp + fn)
            prec.append(p)
        scores.append(np.mean(prec))
    return np.mean(scores)


def inference(image, model_path, **model_params):
    img = image

    model_inference = models.CellposeModel(gpu=True, pretrained_model=model_path)
    preds, flows, _ = model_inference.eval(img, **model_params)

    print(preds.shape)
    print(flows.shape)
    return preds, flows


if __name__ == "__main__":

    st.title("Sartorius Cell Segmentation")

    img = st.file_uploader(label="Upload neuronal cell image")
    model_params = {
        "diameter": 19.0,
        "channels": [0, 0],
        "augment": True,
        "resample": True,
    }
    preds, flows = inference(
        image=img,
        model_path="cellpose_residual_on_style_on_concatenation_off_fold1_ep_649_cv_0.2834",
        **model_params
    )
