import os, gc
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pycocotools.coco import COCO
from PIL import Image


def sanity_check_conversion(cellpose_dir_path, annotation_file_dir, mode, fold):
    # sanity checking conversion
    dir_path = os.path.join(f"{cellpose_dir_path}/{mode}", f"fold_{fold}")
    img_id = random.choice(os.listdir(dir_path)).split("_")[0]
    print(f"Mode: {mode}")
    print(f"Img id : {img_id}")
    tmp_file_orig = Image.open(
        os.path.join(f"{cellpose_dir_path}/{mode}", f"fold_{fold}/{img_id}_img.tif")
    )
    tmp_file_mask = Image.open(
        os.path.join(f"{cellpose_dir_path}/{mode}", f"fold_{fold}/{img_id}_masks.tif")
    )

    orig_arr = np.array(tmp_file_orig)
    print(f"Image shape: {orig_arr.shape}")
    print(f"Image dtype: {orig_arr.dtype}")
    mask_arr = np.array(tmp_file_mask)
    print(f"Mask shape: {mask_arr.shape}")
    print(f"Mask dtype: {mask_arr.dtype}")

    if mode == "train":
        annFile = Path(
            os.path.join(
                annotation_file_dir,
                f"train_annotations/train_fold_{fold}_annotations.json",
            )
        )
    elif mode == "val":
        annFile = Path(
            os.path.join(
                annotation_file_dir, f"val_annotations/val_fold_{fold}_annotations.json"
            )
        )
    coco = COCO(annFile)
    imgIds = coco.getImgIds()
    annIds = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(annIds)
    _, axs = plt.subplots(1, 3, figsize=(20, 16))
    axs[0].imshow(orig_arr, cmap="gray")
    axs[1].imshow(orig_arr, cmap="gray")
    axs[2].imshow(mask_arr, cmap="inferno")
    plt.sca(axs[1])
    coco.showAnns(anns, draw_bbox=False)
    plt.savefig("./sanity_check_data.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cellpose_data_path",
        type=str,
        help="cellpose format data root directory path",
    )
    parser.add_argument(
        "--annotations_path", type=str, help="annotations root directory path"
    )
    parser.add_argument("--mode", type=str, help="train or validation data")
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    # check conversion
    sanity_check_conversion(
        cellpose_dir_path=args.cellpose_data_path,
        annotation_file_dir=args.annotations_path,
        mode=args.mode,
        fold=args.fold,
    )
    # gc.collect()

    # del sanity_check_conversion
