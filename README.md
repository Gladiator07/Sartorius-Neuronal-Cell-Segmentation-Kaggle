# Sartorius-cell-segmentation-kaggle

# Quick Links
- [GitHub](https://github.com/Gladiator07/Sartorius-cell-segmentation-kaggle)
## Trackers
- [Weights and Biases dashboard](https://wandb.ai/gladiator/Sartorius-cell-segmentation-kaggle)
- [Google Sheet tracker](https://docs.google.com/spreadsheets/d/1FOldQ3U40-S6AxmG8E7OOhpDFu27HWGYuLXMx_PU-lU/)
- [Notion Project manager](https://www.notion.so/atharvaingle/Sartorius-Cell-Instance-Segmentation-0199366f6605439c99bdbfa83aaa7796)
---

## Trainer VM
- [Colab VM1](https://colab.research.google.com/drive/1SkjePosIuyQENLPNRbWd9A9E3kOFTa9M)
- [Colab VM2](https://colab.research.google.com/drive/1Y4lfBptVbcZUHfZTbvdnv_z_G3c4R7wV)


- [Kaggle VM](https://www.kaggle.com/atharvaingle/sartorius-detectron2-train-pipeline)
- [Inference V5 (current)](https://www.kaggle.com/atharvaingle/sartorius-inference-v5)

<details>
<summary>Important notebooks/code</summary>

- [First look EDA](https://www.kaggle.com/atharvaingle/sartorius-first-look)
- [(Previous experiment) COCO style json annotations preparation](https://www.kaggle.com/atharvaingle/coco-formated-annotations-prepare)
- [Optimal per class threshold calculator](https://www.kaggle.com/atharvaingle/optimal-threshold-calculator)
- [New 5-fold data (current)](https://www.kaggle.com/atharvaingle/sartorius-kfold-coco-annotations-prepare) --> [Dataset](https://www.kaggle.com/atharvaingle/sartorius-5fold-annots-pct)
- [Single train test split annotations](https://www.kaggle.com/atharvaingle/sartorius-single-split-coco-annotations-prepare) --> [Dataset]()
</details>

---
# Change log + To-Do

### V6 pipeline [cellpose](https://github.com/MouseLand/cellpose) (in progress)🚧🚧🚧
- [x] Create 5-fold data and save in a Kaggle dataset
- [x] Find a way to calculate validation score wrt competition metric
- [x] Add sgd flag for training
- [ ] Add cv calculation after every 50 epochs (3 mins per evaluation phase)
- [ ] Save best models according to the best validation score (in specified directory)
- [ ] Change source code accordingly to add support for mixed precision, model saving, loss logging, etc
- [ ] Add logging to Google sheet and * wandb (* if possible)
- [x] Add oof predictions saving code
- [ ] Figure out how to ensemble cellpose models

### 07/12/2021 (Sartorius V5 pipeline released 🚀🚀🚀)
- Pipeline is completely environment agnostic (can run on colab pro, kaggle, paperspace, jarvislabs.ai) without changing a single line of code
- Supports two mode of validation split - `train_test_split`, `5_fold_split` (training any of these can be invoked by just changing the flag `--val_split` in command line arguments)
- Optimal threshold per class integrated with options to customize from argparse flags (`--start_threshold`, `--threshold_step`, `--score_per_pixel`)
- Calculation of final CV score added (same post-processing as would be done while inferencing)
- Automated writing important information about the experiment to Google Sheet
- Also added storing of `metadata.json` for storing important information about the experiment
- Automated uploading of artifacts to Kaggle based on `--val_split` used in `upload_artifacts_to_kaggle.py` (artifacts folder named as `{experiment_name}_{cv}`
- Complete training pipeline can be invoked by a bash script generated in jupyter notebook environment by specified config file and other misc configuration
- Multiple experiments can be lined up by another bash script by adding multiple train bash scripts generated from jupyter notebook environment

### 20/12/2021 (v3 pipeline)
- Pipeline completely ported to scripts, it's now completely **hardware** and **infrastructure agnostic**
- Main entry point to run the whole pipeline: `python3 train.py --optional arguments from argparse`
- It can automatically detect the infrastructure [colab, kaggle, jarvislabs] and train the model accordingly
- Every possible thing is logged to wandb and saved for further analyzing

#### [Previous notebook based pipeline](https://colab.research.google.com/drive/1nrJTLnVbPal6VcGrcbsp6auHNMj-D40C#scrollTo=181a2292)

<details>
<summary>DONE</summary>
- [x] Clean up redundant artifacts dataset on kaggle and wandb runs generated while testing the code
- [x] Test V5 pipeline on all platforms
- [x] Create `metadata.json` while training the model which will have important information [model_name, fold, optimal_thresholds, scores across different thresholds (pandas dataframe),  cv_raw, cv_with_pp (as close as possible while inferencing), experiment_name, comment]
- [x] Use a bash script to launch experiments
- [x] Add a flag to choose annotations to use (SB split or 5-fold annotations)
- [x] Make the `threshold_per_class.py` independent (will be ran in bash script)
- [x] Add per pixel scoring in evaluation phase as well (optimal threshold + calculate cv with pp)
- [x] Add code to remove .ipynb checkpoints from kaggle upload artifacts dir
- [x] Clean-up pipeline and redundant code (add some extra stuff if required)
- [x] Create detectron 0.6 dataset to use on kaggle for inference (same version you are using for training)
- [x] Add slack alerts to notify on mobile after training is finished and also to notify some metrics
- [x] Optimize the complete pipeline and make it more streamlined (see if bash script can be used for running all 5 folds)
- [x] Following things will be passed to bash script
  - [x] train_file_path
  - [x] annotations to use (5 fold or tts)
  - [x] debug (bool flag)
  - [x] environ
  - [x] model name (from detectron2, str)
  - [x] experiment name (for wandb, file saving, str)
  - [x] threshold per class (bool flag)
  - [x] threshold step (float, need to find a way to pass float args to bash script)
  - [x] upload_artifacts_to_kaggle (bool flag, will invoke another shell script)
  - [x] comment (str)
- [x] Line up all files to run in a bash script (will be written in jupyter environ using %%writefile magic)
  - [x] Files to line up:
    - [x] Train file
    - [x] Threshold per class calculator file
    - [x] CV calculation file (with post-process)
    - [x] Upload model to kaggle shell script
- [x] Find a way to clear logger state or handlers to avoid double print statements in multiple fold runs

#### Completed
- [x] Create new 5-fold data (all in single environment)
- [x] Fix the data preparation classes order
- [x] Write an inference script (need to decide on this)
- [x] Integrate optimal threshold and min pixels calculator in the pipeline (may take extra 15 mins to run)
- [x] Add code to save other configuration like model name, experiment name, optimal thresholds, optimal min pixels in a .pkl or .json file and upload this file as well to artifacts kaggle
- [x] Make the inference code completely automated as your training pipeline, just need to change the experiment name everything else will be automated

- [x] Look once more time into COCO dataset formation code and see if you are doing any mistake there (I highly suspect this because of low scores)
- [x] Create 5 fold data and use it for further experimentations
- [x] Update code wrt new data and add fold as argparse argument
- [x] Update inference script according to new pipeline
- [x] Test new pipeline of jarvislabs
- [x] Create new 5-fold data according to previous time inefficient pipeline (cause I suspect data conversion is the problem for such low scores even with solid lr schedule strategy)
- [x] Integrate CV calculation with optimal thresholds and min pixels in the training pipeline (to be executed after calculating optimal thresholds)
- [x] Automate experiment info writing to google sheet using a script
- [x] Log following things to google sheet
    - Date (within code)
    - Comment (from argparse)
    - Model name
    - fold
    - Experiment (from argparse)
    - CV (raw) (within code)
    - CV (with pp) (within code)
    - Public LB (manually)
    - Configuration (config file link) (within code)
    - Sub link (manually)
    - Environment (from argparse)
- [x] Document all the functions and scripts
- [x] Lower the step size for optimal threshold calculation (might take more time though for training)
- [x] Add debug flag, if true set iters low, disable optimal threshold calculation and disable artifacts upload to kaggle
- [x] Integrate per pixel scoring for evaluation phase
- [x] Try to make CV as much as possible close to LB (by integrating all the preprocessing you are doing while inferencing in CV calculation)
- [ ] ~~Integrate oof predictions saving to a csv file for cv calculation while ensembling~~ (not feasible currently)
</details>

---
# Sartorius-Neuronal-Cell-Segmentation-Kaggle
