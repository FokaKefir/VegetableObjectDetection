# Vegetable Object Detection

Train a Faster R-CNN model on a vegetable dataset from a `dataset.zip` file, or run a simple local web app to load a saved `.pth` model and predict on uploaded images.

## Download Dataset

### Linux/macOS
## Run Web App
bash download_dataset.sh ./data
Start the site on `http://localhost:8000`:

### Windows (PowerShell)
docker run --rm -p 8000:8000 ^
  -v %cd%/models:/models ^
  -e MODEL_PATH=/models/model.pth ^
  vegetable-detector
docker build -t vegetable-detector .
```
You can also upload a `.pth` file directly in the page if you do not want to mount a model path.

## Train From Docker

Put `dataset.zip` in `./data/` and run:

```bash
docker run --rm ^
  -v %cd%/data:/data ^
  -v %cd%/outputs:/outputs ^
  vegetable-detector ^
  main.py ^
  --dataset-zip /data/dataset.zip ^
  --output-dir /outputs ^
  --dataset-percent 50 ^
  --no-preload ^
  --epochs 3
```


## Run
- `main.py` is the training entrypoint.
- `app.py` starts the web UI.
- `--dataset-percent` controls how much of the dataset is used.
- Use `--preload` instead of `--no-preload` if you want to keep images in RAM.
- Model checkpoints are saved to the output folder.
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
