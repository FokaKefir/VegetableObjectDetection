# Vegetable Object Detection

Train a Faster R-CNN model on a vegetable dataset from a `dataset.zip` file.

## Download Dataset

### Linux/macOS
```bash
bash download_dataset.sh ./data
```

### Windows (PowerShell)
```powershell
powershell -ExecutionPolicy Bypass -File download_dataset.ps1 -OutputDir ".\data"
```

This downloads `dataset.zip` from Kaggle.

## Build

```bash
docker build -t vegetable-detector .
```

## Run

Put `dataset.zip` in `./data/` and run:

```bash
docker run --rm ^
  -v %cd%/data:/data ^
  -v %cd%/outputs:/outputs ^
  vegetable-detector ^
  --dataset-zip /data/dataset.zip ^
  --output-dir /outputs ^
  --dataset-percent 50 ^
  --no-preload ^
  --epochs 3
```

```bash
docker run --rm \
  -v "$PWD":/data \
  -v "$PWD/outputs":/outputs \
  vegetable-detector \
  --dataset-zip /data/dataset.zip \
  --output-dir /outputs \
  --dataset-percent 20 \
  --epochs 3
```


## Notes

- `--dataset-percent` controls how much of the dataset is used.
- Use `--preload` instead of `--no-preload` if you want to keep images in RAM.
- Model checkpoints are saved to the output folder.
