from pathlib import Path

import requests


def download(url: str, filename: str) -> Path:
    download_path = Path(".data")
    download_path.mkdir(exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(download_path / filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=4096):
                f.write(chunk)
    return download_path / filename
