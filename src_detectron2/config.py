import os
from pathlib import Path
from argparse import Namespace

# Function to setup paths based on vm
def get_data_config(args: Namespace, fold=0) -> dict:
    """
    Setup paths according to environment and validation split selected

    Args:
        args (Namespace): passed command line arguments to ArgumentParser
        fold (int): fold to validate on (Defaults to 0).

    Returns:
        dict: A dictionary having all the necessary paths
    """

    print(f"[INFO] Setting paths as per the current environment -> {args.environ}")
    if args.environ == "local":
        ROOT_DIR_DATA = Path("/mnt/d/Kaggle/Sartorius_neuronal_cell_segmentation/input")
        ROOT_DIR_ARTIFACTS = Path(
            "D:/Kaggle/Sartorius_neuronal_cell_segmentation/artifacts"
        )

    elif args.environ == "colab":
        ROOT_DIR = Path("/content/Sartorius-cell-segmentation-kaggle")
        ROOT_DIR_DATA = Path("/content/Sartorius-cell-segmentation-kaggle/input")
        ROOT_DIR_ARTIFACTS = Path(f"/content/artifacts/{args.experiment_name}")
        ROOT_DIR_ANNOT_DATA = ROOT_DIR_DATA

    elif args.environ == "kaggle":
        ROOT_DIR = Path("../input/sartorius-code")
        ROOT_DIR_DATA = Path("../input/sartorius-cell-instance-segmentation")
        ROOT_DIR_ARTIFACTS = Path(f"/kaggle/working/artifacts/{args.experiment_name}")

        # change this after preparing new coco format data
        if args.val_split == "tts_split":
            ROOT_DIR_ANNOT_DATA = Path("../input/sartorius-sb-annotations")
        elif args.val_split == "5fold_split":
            ROOT_DIR_ANNOT_DATA = Path("../input/sartorius-5-fold-annot")

    elif args.environ == "jarvislabs":
        ROOT_DIR = Path("/home/Kaggle_comp/Sartorius-cell-segmentation-kaggle")
        ROOT_DIR_DATA = Path(
            "/home/Kaggle_comp/Sartorius-cell-segmentation-kaggle/input"
        )
        ROOT_DIR_ARTIFACTS = Path(f"/home/Kaggle_comp/artifacts/{args.experiment_name}")
        ROOT_DIR_ANNOT_DATA = ROOT_DIR_DATA

    elif args.environ == "paperspace":
        ROOT_DIR = Path("/notebooks/Sartorius-cell-segmentation-kaggle")
        ROOT_DIR_DATA = Path("/notebooks/Sartorius-cell-segmentation-kaggle/input")
        ROOT_DIR_ARTIFACTS = Path(f"/notebooks/artifacts/{args.experiment_name}")
        ROOT_DIR_ANNOT_DATA = ROOT_DIR_DATA

    data_path_config = {
        "root_dir": ROOT_DIR,
        "root_dir_annot_data": ROOT_DIR_ANNOT_DATA,
        "root_dir_data": ROOT_DIR_DATA,
        "root_dir_artifacts": ROOT_DIR_ARTIFACTS,
        "train_csv_path": os.path.join(ROOT_DIR_DATA, "train.csv"),
        # ------------------------------
        # Annotations paths
        # -------------------------------
        # HACK: only one type of annotation's path would be correctly set at a time in case of kaggle environment
        # 5 fold annotations path (current: time inefficient coco conversion, change according to new data creation)
        "train_5fold_annotations": os.path.join(
            ROOT_DIR_ANNOT_DATA,
            f"train_annotations/train_fold_{fold}_annotations.json",
        ),
        "val_5fold_annotations": os.path.join(
            ROOT_DIR_ANNOT_DATA,
            f"val_annotations/val_fold_{fold}_annotations.json",
        ),
        # val train_test_split annotations (SB annots)
        "train_tts_sb_annotations": os.path.join(
            ROOT_DIR_ANNOT_DATA, f"train_tts_sb_annotations.json"
        ),
        "val_tts_sb_annotations": os.path.join(
            ROOT_DIR_ANNOT_DATA, f"val_tts_sb_annotations.json"
        ),
        "sub_csv_path": os.path.join(ROOT_DIR_DATA, "sample_submission.csv"),
        "train_img_dir": os.path.join(ROOT_DIR_DATA, "train"),
        "test_img_dir": os.path.join(ROOT_DIR_DATA, "test"),
        "train_semi_supervised_dir": os.path.join(
            ROOT_DIR_DATA, "train_semi_supervised"
        ),
        "livecell_dataset_2021": os.path.join(ROOT_DIR_DATA, "LIVECell_dataset_2021"),
    }

    return data_path_config
