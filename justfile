# Install dependencies using uv
install:
    uv sync

# Compile locked requirements.txt
to-requirements-txt:
    uv pip compile pyproject.toml -o requirements.txt

# Download MIT-BIH Arrhythmia Database if not already present
download-dataset:
    wget -r -N -c -np https://physionet.org/files/mitdb/1.0.0/ -P dataset

# Run the complete data preparation pipeline (filtering, segmentation, normalization, export)
prepare-data:
    uv run python src/prepare_data.py
