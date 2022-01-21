import os
import gc
import time
import json
import torch
import wandb
import shutil
import argparse
import subprocess
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict
from pprint import pprint
from wandb import AlertLevel
from datetime import timedelta

# detectron2 deps
import detectron2
import detectron2.engine.hooks as hooks
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.datasets import register_coco_instances

# local modules
import secrets
from engine import Trainer
from config import get_data_config
from utils import (
    str2bool,
    copy_file_to_wandb_dir,
    asHours,
    remove_files,
    write_to_google_sheet,
)
from threshold_per_class import calculate_optimal_threshold, ths_cal_setup
from calculate_cv_with_pp import calculate_final_cv


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--environ", type=str, default="colab", required=True)
    parser.add_argument(
        "--val_split",
        type=str,
        default="tts_split",
        help="validation split to use",
        choices=["tts_split", "5fold_split"],
        required=True,
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="custom modified yaml config file path",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="model name from detectron2's model zoo",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="experiment name, this will be used to name files, dirs and wandb grouped run",
    )

    # threshold calculator args
    parser.add_argument(
        "--threshold_step",
        type=float,
        required=True,
        default=0.05,
        help="interval to calculate thresholds",
    )
    parser.add_argument(
        "--start_threshold",
        type=float,
        required=False,
        default=0.05,
        help="Threshold value to start from",
    )
    parser.add_argument(
        "--score_per_pixel",
        type=str2bool,
        required=False,
        default=False,
        help="Whether to overwrite detectron2's functionality (generate per object confidence scores) to generate per pixel scores",
    )
    parser.add_argument("--comment", type=str, required=True)
    parser.add_argument(
        "--fold",
        type=int,
        required=False,
        help="fold to train data on while using 5fold split (currently using time inefficient converted 5 fold split annotations",
    )
    args = parser.parse_args()
    pprint(vars(args))
    return args


def setup(args: argparse.ArgumentParser, data_cfg: dict) -> Dict:
    """
    Initial setup (registering COCO instances, config file merging)

    Args:
        args (ArgumentParser): parsed arguments
        data_cfg (dict): dictionary having all necessary paths

    Returns:
        Dict: configuration
    """
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
    # Merging from specified model config file
    # https://detectron2.readthedocs.io/en/latest/modules/model_zoo.html#detectron2.model_zoo.get_config_file
    cfg.merge_from_file(model_zoo.get_config_file(args.model_name))

    # Merging modified config to complete file
    cfg.merge_from_file(args.config_path, allow_unsafe=True)

    # just for sanity check (#TODO: Remove this after figuring out the pipeline)
    cfg.INPUT.MASK_FORMAT = "bitmask"
    return cfg


if __name__ == "__main__":

    print(f"\n\n[INFO] USING DETECTRON2 VERSION --> {detectron2.__version__}")
    args = parse_args()

    # some sanity checks for `val_split`
    if args.val_split == "tts_split" and args.fold is not None:
        raise ValueError(
            "tts_split and fold flags can't be used at the same time",
            "switch --val_split flag from tts_split to 5fold_split to use fold argument",
        )
    if args.val_split == "5fold_split" and args.fold is None:
        raise ValueError(
            "5fold split requires fold flag to be specified",
            "add --fold flag with an integer (from 0 to 4) to validate the model on",
        )

    print(f"[INFO] VALIDATION SPLIT --> {args.val_split}")
    os.environ["WANDB_API_KEY"] = secrets.WANDB_API_KEY
    os.environ["WANDB_SILENT"] = secrets.WANDB_SILENT

    exp_start_time = time.time()

    # TODO: Remove this line if not required
    # seed_everything(seed=args.seed)

    ###############################################
    # Setting paths, creating dirs
    ###############################################
    data_cfg = get_data_config(args=args, fold=args.fold)

    # artifacts dir (models, config, etc) [will have the same name as experiment_name]
    if args.val_split == "tts_split":
        artifacts_dir = data_cfg["root_dir_artifacts"]
    elif args.val_split == "5fold_split":
        artifacts_dir = os.path.join(
            data_cfg["root_dir_artifacts"], f"fold_{args.fold}"
        )

    os.makedirs(data_cfg["root_dir_artifacts"], exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    print(f"[INFO] Models, config, metrics, metadata will be saved to {artifacts_dir}")

    ################################################
    # Setting configuration
    ################################################

    # setting training & model config
    cfg = setup(args, data_cfg)
    cfg.OUTPUT_DIR = str(artifacts_dir)

    # HACK: this is done because `os.cpu_count()`
    # returns 64 in jarvislabs env which freezes data loading
    if args.environ == "jarvislabs":
        cfg.DATALOADER.NUM_WORKERS = 7
    else:
        cfg.DATALOADER.NUM_WORKERS = os.cpu_count()  # for colab, kaggle, paperspace

    print(f"[INFO] Using {cfg.DATALOADER.NUM_WORKERS} workers for data loading")

    # debug flag customizations
    if args.debug:
        cfg.WARMUP_ITERS = 100
        cfg.SOLVER.MAX_ITER = 200
        args.step = 0.5

    # Custom paths
    modified_config_file_path = os.path.join(
        data_cfg["root_dir_artifacts"], f"{args.experiment_name}_modified_cfg.yaml"
    )
    complete_config_file_path = os.path.join(
        data_cfg["root_dir_artifacts"], f"{args.experiment_name}_cfg.yaml"
    )
    log_file_path = os.path.join(
        artifacts_dir,
        f"{args.experiment_name}_fold_{args.fold}.log"
        if args.val_split == "5fold_split"
        else f"{args.experiment_name}.log",
    )
    threshold_scores_df_path = os.path.join(
        cfg.OUTPUT_DIR, "threshold_per_class_scores.csv"
    )

    # save modified config file
    shutil.copyfile(args.config_path, modified_config_file_path)
    # save complete config file
    with open(complete_config_file_path, "w") as f:
        f.write(cfg.dump())

    # setup detectron2 logger
    setup_logger(output=log_file_path)

    # wandb init
    wandb.init(
        project="Sartorius-cell-segmentation-kaggle",
        entity="gladiator",
        group=args.experiment_name,
        name="tts_run" if args.val_split == "tts_split" else f"fold_{args.fold}",
        sync_tensorboard=True,
    )

    ############################################
    # TRAINER
    ############################################
    trainer = Trainer(cfg)

    # best checkpoint hook
    checkpointer = DetectionCheckpointer(
        trainer.model, save_dir=cfg.OUTPUT_DIR, save_to_disk=True
    )
    best_checkpoint_hook = hooks.BestCheckpointer(
        eval_period=cfg.TEST.EVAL_PERIOD,
        checkpointer=checkpointer,
        val_metric="map_iou",
        mode="max",
    )
    trainer.register_hooks([best_checkpoint_hook])

    trainer.resume_or_load(resume=False)
    trainer.train()

    # getting the best map_iou score
    metrics_df = pd.read_json(
        os.path.join(f"{artifacts_dir}", f"metrics.json"),
        orient="records",
        lines=True,
    )
    mdf = metrics_df.sort_values("iteration")
    best_score = round(mdf["map_iou"].max(), 4)
    best_score_idx = mdf["map_iou"].idxmax()

    print(
        f"\n\n============== CV score (without post-processing) -> {best_score:.5f} (obtained @ iteration {mdf.iloc[best_score_idx]['iteration']}) =============="
    )

    # TODO: Add/remove logging sample predictions to wandb (will be removed mostly)
    # print("\n[INFO] Logging sample predictions to wandb ....")
    # log_val_predictions_wandb(cfg, n_images=20)

    wandb.log({"cv": best_score})

    torch.cuda.empty_cache()
    gc.collect()

    # alert run info and CV to wandb
    fold_info = f"fold-{args.fold}" if args.val_split == "5fold_split" else "tts_split"
    wandb.alert(
        title=f"{args.experiment_name}, {fold_info}",
        text=f"CV (without post-process) --> {best_score}",
        level=AlertLevel.INFO,
        wait_duration=timedelta(seconds=0),
    )
    train_end_time = time.time()
    total_train_elapsed_time = train_end_time - exp_start_time

    print(
        f"\n\n-------------- Training for {args.experiment_name} completed successfully in {asHours(total_train_elapsed_time)} --------------\n\n"
    )
    ths_cal_start_time = time.time()
    print("\n\n" + "=" * 15 + " Calculating optimal threshold per class " + "=" * 15)

    # ##################################################################
    # Calcuating thresholds per class and final CV with post-processing
    # #################################################################

    # initial setup for threshold calculation
    ths_cfg = ths_cal_setup(
        args,
        data_cfg,
        model_weights_path=os.path.join(cfg.OUTPUT_DIR, "model_best.pth"),
    )
    ths_scores_df, optimal_thresholds = calculate_optimal_threshold(
        cfg=ths_cfg,
        start_threshold=args.start_threshold,
        threshold_step=args.threshold_step,
        score_per_pixel=args.score_per_pixel,
        calculate_optimal_min_pixels=False,
    )
    # saving threshold_per_class_scores to dataframe
    ths_scores_df.to_csv(threshold_scores_df_path, index=False)

    # -------------------------------------------------------
    # Calculating final CV with post-process as would be done while inferencing
    # ----------------------------------------------------
    print(f"[INFO] Calculating final CV score with post-process ...")

    final_cv_with_pp = calculate_final_cv(
        args,
        model_weights_path=os.path.join(cfg.OUTPUT_DIR, "model_best.pth"),
        optimal_thresholds=optimal_thresholds,
    )
    final_cv_with_pp = np.round(final_cv_with_pp, 4)

    wandb.log({"cv_with_pp": final_cv_with_pp})

    # alert run info and CV to wandb
    fold_info = f"fold-{args.fold}" if args.val_split == "5fold_split" else "tts_split"
    wandb.alert(
        title=f"{args.experiment_name}, {fold_info}",
        text=f"CV (with post-process) --> {final_cv_with_pp}",
        level=AlertLevel.INFO,
        wait_duration=timedelta(seconds=0),
    )

    # ---------------------------------------------------------------
    # Saving important data about experiment to metadata.json file
    # ---------------------------------------------------------------
    # wandb run url
    run_url = wandb.run.get_url()
    metadata_dict = {
        "environ": args.environ,
        "wandb_run_link": run_url,
        "configuration": f"{run_url}/files/{args.experiment_name}_modified_cfg.yaml",
        "model_name": args.model_name,
        "experiment_name": args.experiment_name,
        "fold": "tts_split" if args.val_split == "tts_split" else args.fold,
        "cv": best_score,
        "cv_with_pp": final_cv_with_pp,
        "threshold_per_class": optimal_thresholds,
        "comment": args.comment,
    }

    # save metadata_dict to metadata.json
    metadata_file_path = os.path.join(cfg.OUTPUT_DIR, "metadata.json")
    metadata_json_obj = json.dumps(metadata_dict, indent=4)
    with open(metadata_file_path, "w") as outfile:
        outfile.write(metadata_json_obj)

    # logging all of the information to google sheet
    final_meta_file = open(metadata_file_path, "r")
    metadata = json.load(final_meta_file)
    write_to_google_sheet(
        credential_file_path=os.path.join(
            data_cfg["root_dir"], "scripts/credentials.json"
        ),
        metadata_file_obj=metadata,
    )
    print("Done.")
    #####################################################################################
    # Final bits of cleaning (uploading models to wandb, kaggle, deleting, etc)
    # ##################################################################################

    # Save models, logs to wandb
    files_to_upload_wandb = {
        # log file
        log_file_path: f"{args.experiment_name}_fold_{args.fold}.log"
        if args.val_split == "5fold_split"
        else f"{args.experiment_name}.log",
        # modified config file
        modified_config_file_path: f"{args.experiment_name}_modified_cfg.yaml",
        # complete config file
        complete_config_file_path: f"{args.experiment_name}_cfg.yaml",
        # metrics file
        os.path.join(
            cfg.OUTPUT_DIR, "metrics.json"
        ): f"{args.experiment_name}_fold_{args.fold}_metrics.json"
        if args.val_split == "5fold_split"
        else f"{args.experiment_name}_metrics.json",
        # threshold_per_class_scores df
        threshold_scores_df_path: f"{args.experiment_name}_fold_{args.fold}_threshold_per_class_scores.csv"
        if args.val_split == "5fold_split"
        else f"{args.experiment_name}_threshold_per_class_scores.csv",
        # final model  (# TODO: remove logging final model after a solid pipeline)
        # os.path.join(
        #     cfg.OUTPUT_DIR, "model_final.pth"
        # ): f"{args.experiment_name}_fold_{args.fold}_model_final.pth"
        # if args.val_split == "5fold_split"
        # else f"{args.experiment_name}_model_final.pth",
        # best model
        os.path.join(
            cfg.OUTPUT_DIR, "model_best.pth"
        ): f"{args.experiment_name}_fold_{args.fold}_model_best.pth"
        if args.val_split == "5fold_split"
        else f"{args.experiment_name}_model_best.pth",
        # metadata
        metadata_file_path: f"{args.experiment_name}_fold_{args.fold}_metadata.json"
        if args.val_split == "5fold_split"
        else f"{args.experiment_name}_metadata.json",
    }

    for key, value in tqdm(
        files_to_upload_wandb.items(),
        total=len(files_to_upload_wandb),
        desc="Copying files to wandb run dir",
    ):
        copy_file_to_wandb_dir(key, value)

    exp_end_time = time.time()
    ths_cal_time = exp_end_time - ths_cal_start_time
    total_exp_time = exp_end_time - exp_start_time

    print("\n\n" + "=" * 10 + f" CV without post-process : {best_score} " + "=" * 10)
    print(
        "\n"
        + "=" * 10
        + f" Final CV with post-process : {final_cv_with_pp} "
        + "=" * 10
    )
    print(
        f"\n\n-------------- Threshold calculation + final CV completed successfully in {asHours(ths_cal_time)} --------------\n\n"
    )
    print(
        f"-------------- Total experiment {args.experiment_name} completed successfully in {asHours(total_exp_time)} --------------\n\n"
    )

    wandb.log({"experiment_time": total_exp_time})
    ########################
    # Cleaning up
    ########################
    print("Uploading artifacts to wandb ...")
    wandb.finish()
    print("Done.")
    torch.cuda.empty_cache()
    gc.collect()

    print(f"[INFO] Cleaning up redundant files")
    files_to_keep = [
        "model_best.pth",
        f"metrics.json",
        f"optimal_thresholds.json",
        f"metadata.json",
        f"{args.experiment_name}_cfg.yaml",
        f"{args.experiment_name}_modified_cfg.yaml",
    ]

    remove_files(files_to_keep, data_dir=cfg.OUTPUT_DIR)

    print(f"[INFO] Finished {args.experiment_name} train run")
    print("=" * 100)

    subprocess.run(f"rm -rf {cfg.OUTPUT_DIR}/.ipynb_checkpoints".split())

    gc.collect()
