# Web App

Simple local web UI for loading a saved `.pth` model and running predictions on uploaded images.

## Create a virtual environment

From the repository root:

### Windows PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux/macOS
```bash
python -m venv .venv
source .venv/bin/activate
```

## Install web dependencies

```bash
pip install -r web/requirements.txt
```

## Run the web app

```bash
python web/app.py
```

Open `http://localhost:8000` in your browser.

## Load a model

You can either:

- set `MODEL_PATH` to a local `.pth` file before starting the app
- upload a `.pth` file from the page

Example:

### Windows PowerShell
```powershell
$env:MODEL_PATH = "C:\models\model.pth"
python web/app.py
```

### Linux/macOS
```bash
export MODEL_PATH=/models/model.pth
python web/app.py
```
