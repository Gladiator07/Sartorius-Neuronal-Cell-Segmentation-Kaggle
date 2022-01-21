echo "Writing Kaggle API key to ~/.kaggle/kaggle.json"

mkdir -p ~/.kaggle
cat <<EOF > ~/.kaggle/kaggle.json
{"username":"<your_username_here>","key":"<your_kaggle_key_here>"}
EOF
cat ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
echo "Kaggle API Key successfully linked !!!"
pip3 install --upgrade --force-reinstall kaggle
pip3 install --upgrade wandb
pip3 install pycocotools
pip3 install --upgrade --force-reinstall gspread
# cellpose
pip3 uninstall -y -q yellowbrick
pip3 install -q tifffile
pip3 install -q folium==0.2.1
pip3 install -q imgaug==0.2.5
pip3 install -q cellpose 
pip3 install -q wget
pip3 install -q memory_profiler
pip3 install -q fpdf
pip3 install --upgrade --force-reinstall numpy

# detectron2
# pip3 install 'git+https://github.com/facebookresearch/detectron2.git'

cd ~/.
mkdir Kaggle_comp
cd Kaggle_comp/
# setup code from GitHub


# Download data
cd Sartorius-cell-segmentation-kaggle/
mkdir input/
cd input/
kaggle competitions download -c sartorius-cell-instance-segmentation
unzip sartorius-cell-instance-segmentation.zip
rm sartorius-cell-instance-segmentation.zip

# --------------------------------
# Annotations & Cellpose tif files
# --------------------------------

# SB annotations
kaggle datasets download -d atharvaingle/sartorius-sb-annotations
unzip sartorius-sb-annotations.zip
rm sartorius-sb-annotations.zip

# new 5-fold annotations file (using fast pycocotools implementation)
kaggle datasets download -d atharvaingle/sartorius-5fold-annots-pct
unzip sartorius-5fold-annots-pct.zip
rm sartorius-5fold-annots-pct.zip

mkdir cellpose_data/
cd cellpose_data/
# cellpose data
kaggle datasets download -d atharvaingle/cellpose-data
unzip cellpose-data.zip
rm cellpose-data.zip
# fold csv
kaggle datasets download -d atharvaingle/sartorius-fold-csv
unzip sartorius-fold-csv.zip
rm sartorius-fold-csv.zip
echo "Data downloaded and unzipped successfully !!!"
cd ..
