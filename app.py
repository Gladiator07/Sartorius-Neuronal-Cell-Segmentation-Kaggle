import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from cellpose import models


@st.cache()
def load_model(model_path):
    inf_model = models.CellposeModel(gpu=False, pretrained_model=model_path)
    return inf_model


# def inference(img, model_path, **model_params):
#    inf_model = models.CellposeModel(gpu=False, pretrained_model=model_path)
#    preds, flows, _ = inf_model.eval([img], **model_params)
#    return preds, flows


if __name__ == "__main__":

    st.title("Sartorius Neuronal Cell Segmentation")
    model_path = (
        "./cellpose_residual_on_style_on_concatenation_off_fold1_ep_649_cv_0.2834"
    )
    inf_model = models.CellposeModel(gpu=False, pretrained_model=model_path)
    uploaded_img = st.file_uploader(label="Upload neuronal cell image")

    with st.expander("View input image"):
        if uploaded_img is not None:
            st.image(uploaded_img)
        else:
            st.warning("Please upload an image")

    segment = st.button("Perform segmentation")

    if uploaded_img is not None and segment:
        img = Image.open(uploaded_img)
        img = np.array(img)

        model_params = {
            "diameter": 19.0,
            "channels": [0, 0],
            "augment": True,
            "resample": True,
        }
        with st.spinner("Performing segmentation. This might take a while..."):
            # preds, flows = inference(
            #   img=img,
            #  model_path="./cellpose_residual_on_style_on_concatenation_off_fold1_ep_649_cv_0.2834",
            # **model_params
            # )
            preds, flows, _ = inf_model.eval([img], **model_params)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.axis("off")
        ax2.axis("off")
        ax3.axis("off")
        ax1.set_title("Original Image")
        ax1.imshow(img, cmap="gray")
        ax2.set_title("Segmented image")
        ax2.imshow(preds[0])
        ax3.set_title("Image flows")
        ax3.imshow(flows[0][0])
        st.pyplot(fig)
