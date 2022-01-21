# Created in Google Colab Pro (paths might differ)
import os
import numpy as np
import tifffile as tiff

from tqdm import tqdm
from PIL import Image
from pathlib import Path

# pycocotools
from pycocotools.coco import COCO


def convert_coco_to_cellpose(
    annotations_dir, data_dir, mode="train", fold=0, output_path=None
):

    # create directories
    if mode == "train":
        data_dir_out = os.path.join(output_path, f"train/fold_{fold}")
        ann_file_path = os.path.join(
            annotations_dir, f"train_annotations/train_fold_{fold}_annotations.json"
        )

    elif mode == "val":
        data_dir_out = os.path.join(output_path, f"val/fold_{fold}")
        ann_file_path = os.path.join(
            annotations_dir, f"val_annotations/val_fold_{fold}_annotations.json"
        )

    os.makedirs(data_dir_out, exist_ok=True)

    # read annotations
    annFile = Path(ann_file_path)
    coco = COCO(annFile)
    imgIds = coco.getImgIds()
    cat_ids = coco.getCatIds()

    for id in tqdm(imgIds, total=len(imgIds)):
        # load the image
        # [{'id': '0030fd0e6378', 'width': 704, 'height': 520, 'file_name': 'train/0030fd0e6378.png'}]
        img = coco.loadImgs(ids=[id])
        # get annotation ids for the image (total for all images : 73585)
        anns_ids = coco.getAnnIds(imgIds=[id], catIds=cat_ids, iscrowd=None)
        # get the annotations
        anns = coco.loadAnns(anns_ids)  # list
        anns_img = np.zeros((img[0]["height"], img[0]["width"]), dtype=np.uint16)
        for idx, ann in enumerate(anns):
            anns_img = np.maximum(anns_img, coco.annToMask(ann) * (idx + 1))

        anns_img = np.array(anns_img, dtype=np.uint16)

        # saving converted image
        im_path = os.path.join(data_dir_out, f"{id}_masks.tif")
        tiff.imsave(im_path, anns_img)

        # saving original image
        # don't forget to add image filter in arguments
        # to training script
        orig_img_path = os.path.join(data_dir, f"{id}.png")
        orig_img_im = Image.open(orig_img_path)
        orig_img_arr = np.array(orig_img_im, dtype=np.uint16)
        orig_img_converted_path = f"{data_dir_out}/{id}_img.tif"
        tiff.imsave(orig_img_converted_path, orig_img_arr)


if __name__ == "__main__":

    # setting up paths

    # root data dir
    data_dir = Path("/content/Sartorius-cell-segmentation-kaggle/input/train")
    # annotations root dir
    annotations_dir = "/content/Sartorius-cell-segmentation-kaggle/input"

    # root dir to store cell pose format converted data
    cell_pose_data_path = Path("/content/cellpose_data")
    os.makedirs(cell_pose_data_path, exist_ok=True)

    # create 5-fold cellpose data

    for fold in range(5):
        print(f"Creating fold -> {fold} data...")
        # train
        convert_coco_to_cellpose(
            annotations_dir=annotations_dir,
            data_dir=data_dir,
            mode="train",
            fold=fold,
            output_path=cell_pose_data_path,
        )
        # test (val)
        convert_coco_to_cellpose(
            annotations_dir=annotations_dir,
            data_dir=data_dir,
            mode="val",
            fold=fold,
            output_path=cell_pose_data_path,
        )

        print("-" * 60)
    print("Data conversion successfully completed.")
