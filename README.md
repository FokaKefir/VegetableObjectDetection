# Vegetable Object Detection

Train a Faster R-CNN model on a vegetable dataset from a `dataset.zip` file, or run a simple local web app to load a saved `.pth` model and predict on uploaded images.

## Download Assets

### Best Model

### Linux/macOS
```bash
bash download_best_model.sh ./models
```

### Windows (PowerShell)
```powershell
powershell -ExecutionPolicy Bypass -File download_best_model.ps1 -OutputDir ".\models"
```

This downloads `best_model.pth` into `./models`.

### Kaggle Dataset

### Linux/macOS
```bash
bash download_dataset.sh ./data
```

### Windows (PowerShell)
```powershell
powershell -ExecutionPolicy Bypass -File download_dataset.ps1 -OutputDir ".\data"
```

This downloads the older Kaggle `dataset.zip` into `./data`.

### Custom Dataset

### Linux/macOS
```bash
bash download_custom_dataset.sh ./data
```

### Windows (PowerShell)
```powershell
powershell -ExecutionPolicy Bypass -File download_custom_dataset.ps1 -OutputDir ".\data"
```

This downloads `custom_dataset.zip` into `./data`.

## Build Docker Image

```bash
docker build -t vegetable-detector .
```

## Run the Web App

Start the browser UI on `http://localhost:8000`:

### Linux/macOS
```bash
docker run --rm -p 8000:8000 \
  -v "$PWD/models":/models \
  -e MODEL_PATH=/models/model.pth \
  vegetable-detector
```

### Windows (PowerShell)
```powershell
docker run --rm -p 8000:8000 ^
  -v %cd%/models:/models ^
  -e MODEL_PATH=/models/model.pth ^
  vegetable-detector
```

You can also upload a `.pth` file directly in the page.

## Train From Docker

### Linux/macOS
```bash
docker run --rm \
  -v "$PWD/data":/data \
  -v "$PWD/outputs":/outputs \
  vegetable-detector \
  main.py \
  --dataset-zip /data/dataset.zip \
  --output-dir /outputs \
  --dataset-percent 20 \
  --epochs 3
```

### Windows (PowerShell)
```powershell
docker run --rm ^
  -v %cd%/data:/data ^
  -v %cd%/outputs:/outputs ^
  vegetable-detector ^
  main.py ^
  --dataset-zip /data/dataset.zip ^
  --output-dir /outputs ^
  --dataset-percent 20 ^
  --epochs 3
```

## Web App Files

- `web/app.py` runs the Flask web app.
- `web/index.html` contains the UI.
- `web/requirements.txt` lists the web-only Python dependencies.
- `web/README.md` explains how to run the web app in a local `.venv`.

## Notes

- `--dataset-percent` controls how much of the dataset is used.
- `--preload` keeps images in RAM; `--no-preload` loads them from disk.
- Model checkpoints are saved to the output folder.
