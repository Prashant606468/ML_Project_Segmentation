#!/usr/bin/env bash
# PraNet benchmark split (Fan et al., MICCAI 2020), links from https://github.com/DengPingFan/PraNet
#   TrainDataset: 900 Kvasir-SEG + 550 CVC-ClinicDB
#   TestDataset : Kvasir (100), CVC-ClinicDB (62), CVC-ColonDB (380), ETIS-LaribPolypDB (196), CVC-300 (60)
# Also downloads the YOLOv8 detector weights used as backbone.
set -e
cd "$(dirname "$0")/.."
mkdir -p data/raw weights
gdown 1lODorfB33jbd-im-qrtUgWnZXxB94F55 -O data/raw/TrainDataset.zip
gdown 1o8OfBvYE6K-EpDyvzsmMPndnUMwb540R -O data/raw/TestDataset.zip
unzip -q data/raw/TrainDataset.zip -d data
unzip -q data/raw/TestDataset.zip -d data
for v in n s; do
  wget -q -O weights/yolov8$v.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8$v.pt
done
