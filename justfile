download-dataset:
    curl.exe -L -o dataset/mitdb.zip https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip
    tar -xf dataset/mitdb.zip -C dataset

    or 

    wget -r -N -c -np https://physionet.org/files/mitdb/1.0.0/ -P dataset 

    
install-packages:
    uv sync

to-requirements-txt:
    uv pip compile pyproject.toml -o requirements.txt

