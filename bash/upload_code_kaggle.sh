#!/bin/bash
cd ../

printf "Checking code changes in main repo's source code\n"
git pull

printf "\nGoing to full_code directory\n"
cd full_code/Sartorius-cell-segmentation-kaggle

printf "\n Pulling changes to kaggle dataset's code\n"
echo "Pulling latest source code"
git pull

cd ..
kaggle datasets version -p ./ -m "update" --dir-mode tar