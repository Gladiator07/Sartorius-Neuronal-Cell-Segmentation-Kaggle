echo "Writing Kaggle API key to ~/.kaggle/kaggle.json"

mkdir -p ~/.kaggle
cat <<EOF > ~/.kaggle/kaggle.json
{"username":"<your_username_here>","key":"<your_kaggle_key_here>"}
EOF
cat ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
echo "Kaggle API Key successfully linked !!!"
pip3 install pandas
pip3 install --upgrade --force-reinstall --no-deps kaggle
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