""""
THIS SCRIPT IS DEPRECATED
This is an independent module, not included in training pipeline
Just to sanity check if the data is properly processed

To use in colab or any other jupyter notebook environment:
- Add this module to sys.path
- `from sanity_check_data import sanity_check_processed_data`
- Run the imported function -> `sanity_check_processed_data(ENVIRON)`

"""
import gc
import cv2
import matplotlib.pyplot as plt

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances

# local modules
from config import get_data_config


def sanity_check_processed_data(environ):

    data_cfg = get_data_config(environ, fold=0)

    # register dataset in COCO's json annotation format for instance detection
    try:
        register_coco_instances(
            "sartorius_train",
            {},
            data_cfg["train_annotations"],
            data_cfg["root_dir_data"],
        )
        register_coco_instances(
            "sartorius_val", {}, data_cfg["val_annotations"], data_cfg["root_dir_data"]
        )
    except Exception as e:
        print(f"\n{e}")

    # global dict for metadata (class names, etc) [don't abuse this]
    metadata = MetadataCatalog.get("sartorius_train")

    # https://detectron2.readthedocs.io/en/latest/modules/data.html?highlight=MetadataCatalog#detectron2-data
    # global dict that stores information about the datasets and how to obtain them
    train_ds = DatasetCatalog.get("sartorius_train")

    ###########################################################
    # Testing a sample of data
    def visualize_one_sample(train_ds, idx=42):
        d = train_ds[idx]
        img = cv2.imread(d["file_name"])  # (520, 704, 3)
        print(img.shape)
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata)
        out = visualizer.draw_dataset_dict(d)
        plt.figure(figsize=(12, 10))
        plt.imshow(out.get_image()[:, :, ::-1])
        # plt.show()

    visualize_one_sample(train_ds, idx=450)

    gc.collect()
