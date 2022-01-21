import json
import glob
import argparse
import subprocess
import numpy as np

# local modules
from config import get_data_config


if __name__ == "__main__":

    print("\n\n" + "=" * 15 + " Uploading artifacts to Kaggle " + "=" * 15)
    parser = argparse.ArgumentParser()
    parser.add_argument("--environ", type=str, default="colab", required=True)

    parser.add_argument(
        "--experiment_name",
        type=str,
        help="experiment name to determine the current artifacts dir",
    )

    parser.add_argument(
        "--val_split",
        type=str,
        default="tts_split",
        help="validation split to use",
        choices=["tts_split", "5fold_split"],
        required=True,
    )

    args = parser.parse_args()

    data_cfg = get_data_config(args=args)

    code_path = data_cfg["root_dir"]
    artifacts_dir = data_cfg["root_dir_artifacts"]

    if args.val_split == "5fold_split":
        cvs = []
        cvs_with_pp = []
        for file_name in glob.glob(f"{artifacts_dir}/*/*"):
            if "metadata.json" in file_name:
                print(f"Loading cv from {file_name}")
                f = open(file_name, "r")
                j_obj = json.load(f)

                cv = j_obj["cv"]
                cv_with_pp = j_obj["cv_with_pp"]
                cvs.append(cv)
                cvs_with_pp.append(cv_with_pp)
                f.close()

        final_cv = np.mean(cvs)
        final_cv_with_pp = np.mean(cvs_with_pp)
        print(f"CVS -> {cvs}")
        print(f"CVS with post-process -> {cvs_with_pp}")
        print(f"Mean CV -> {final_cv}, Mean CV with post-process -> {final_cv_with_pp}")

    elif args.val_split == "tts_split":
        f = open(f"{artifacts_dir}/metadata.json", "r")
        j_obj = json.load(f)
        final_cv = j_obj["cv"]
        final_cv_with_pp = j_obj["cv_with_pp"]
        f.close()

        print(f"CV --> {final_cv}")
        print(f"CV with post-process --> {final_cv_with_pp}")

    print(f"[INFO] Uploading artifacts to kaggle from {artifacts_dir}")
    subprocess.run(
        f"bash {code_path}/bash/upload_model_to_kaggle.sh {args.experiment_name} {final_cv}".split()
    )
