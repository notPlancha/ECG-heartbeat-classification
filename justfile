# Justfile for ECG Heartbeat Classification on Edge-AI

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

# Execute CRISP-DM Stage 1: Business / Application Understanding notebook
run-task1:
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/Task1_Business_Understanding.ipynb

# Execute CRISP-DM Stage 2: Data Understanding (EDA) notebook
run-task2:
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/EDA.ipynb

run-eda: run-task2

# Execute CRISP-DM Stage 3: Data Preparation notebook
run-task3:
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/Task3_Data_Preparation.ipynb

# Execute all CRISP-DM Submission 1 notebooks in sequence
run-all-notebooks: run-task1 run-task2 run-task3
