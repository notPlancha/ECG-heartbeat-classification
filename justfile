download-dataset:
    wget -r -N -c -np https://physionet.org/files/mitdb/1.0.0/ -P dataset

install-packages:
    uv sync

to-requirements-txt:
    uv pip compile pyproject.toml -o requirements.txt

