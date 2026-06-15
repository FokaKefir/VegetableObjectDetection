# Vegetable Object Detection

Train a Faster R-CNN model on a vegetable dataset from a `dataset.zip` file.

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

## Notes

- `--dataset-percent` controls how much of the dataset is used.
- Use `--preload` instead of `--no-preload` if you want to keep images in RAM.
- Model checkpoints are saved to the output folder.
