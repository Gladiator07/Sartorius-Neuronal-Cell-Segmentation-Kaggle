import os
from PIL import Image
import streamlit as st
import ast
import numpy as np
import pandas as pd
from cellpose import models, io, plot
from pathlib import Path


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


def inference(image, model_path, **model_params):
    img = image

    model_inference = models.CellposeModel(gpu=False, pretrained_model=model_path)
    preds, flows, _ = model_inference.eval(img, **model_params)

    print(preds.shape)
    print(flows.shape)
    return preds, flows


if __name__ == "__main__":

    st.title("Sartorius Cell Segmentation")

    uploaded_img = st.file_uploader(label="Upload neuronal cell image")
    if uploaded_img is not None:
        img = Image.open(uploaded_img)
        st.image(img)

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
