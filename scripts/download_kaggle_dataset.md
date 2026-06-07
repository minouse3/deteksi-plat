# Download Kaggle Dataset

Dataset: `linkgish/indonesian-plate-number-from-multi-sources`

1. Install the Kaggle API inside the virtual environment.

```bash
pip install kaggle
```

2. Place `kaggle.json` in the Kaggle credentials directory.

Windows:

```powershell
mkdir $env:USERPROFILE\.kaggle
copy kaggle.json $env:USERPROFILE\.kaggle\kaggle.json
```

macOS/Linux:

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

3. Download and unzip the dataset.

```bash
kaggle datasets download -d linkgish/indonesian-plate-number-from-multi-sources -p datasets/kaggle_raw --unzip
```

4. Inspect and prepare it.

```bash
python scripts/inspect_dataset.py --root datasets/kaggle_raw
python scripts/prepare_kaggle_dataset.py --raw-root datasets/kaggle_raw
```
