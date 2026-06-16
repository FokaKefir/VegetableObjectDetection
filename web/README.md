# Web App

This folder contains the local Flask app for loading a trained `.pth` checkpoint and running vegetable detection in the browser.

## What’s here

- `app.py` runs the API and serves the UI.
- `index.html` is the browser interface.
- `best_model.pth` is the local checkpoint used by default when present.
- `requirements.txt` lists the web-only Python dependencies.

## Recommended setup

From the repository root:

### Windows PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r web/requirements.txt
```

### Git Bash / Linux / macOS
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r web/requirements.txt
```

## Download the model

The web app looks for `best_model.pth` in this folder by default.

### Git Bash
```bash
../download_best_model.sh
```

### PowerShell
```powershell
..\download_best_model.ps1
```

If you want the file saved somewhere else, pass an output directory:

### Git Bash
```bash
../download_best_model.sh ./models
```

### PowerShell
```powershell
..\download_best_model.ps1 -OutputDir ".\models"
```

## Run the app

From this `web` folder:

```bash
python app.py
```

Open `http://localhost:8000` in your browser.

The app also listens on your LAN address, which is useful for testing from another device on the same network.

## Using a different model path

You can override the default checkpoint path with `MODEL_PATH`.

### Windows PowerShell
```powershell
$env:MODEL_PATH = "C:\models\model.pth"
python app.py
```

### Git Bash / Linux / macOS
```bash
export MODEL_PATH=/models/model.pth
python app.py
```

## Using the UI

1. Load a checkpoint by path or upload a `.pth` file.
2. Choose an image.
3. Click Predict.

The preview now normalizes image orientation, so phone photos should render upright before boxes are drawn.

## Notes

- The dataset download scripts are for training, not for the web viewer.
- If the app starts but no model is loaded, upload a `.pth` file from the page or set `MODEL_PATH` before launching.
- `best_model.pth` and `custom_dataset.zip` are ignored at the repo level, so they stay local by default.
