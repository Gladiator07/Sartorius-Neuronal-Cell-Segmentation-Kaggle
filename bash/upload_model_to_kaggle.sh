echo "Preparing data to upload..."

# data_meta_file=./artifacts/$1/dataset-metadata.json
cat <<EOF > ./artifacts/$1/dataset-metadata.json
{
  "title": "$1_cv_$2",
  "id": "atharvaingle/$1",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}
EOF

kaggle datasets create -p ./artifacts/$1 --dir-mode tar

echo "Artifacts uploaded to kaggle successfully"