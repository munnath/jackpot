import os  
import sys  
import hashlib  
import requests  
from pathlib import Path  
  
URL = "https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth"  
DEFAULT_DIR = "../model_zoo"  
FILENAME = "drunet_color.pth"  
  
# Optional: put the known SHA256 here once you verify it the first time  
KNOWN_SHA256 = None  # e.g., "abc123..."; leave None to skip check  
  
def sha256sum(path: Path) -> str:  
    h = hashlib.sha256()  
    with open(path, "rb") as f:  
        for chunk in iter(lambda: f.read(8192), b""):  
            h.update(chunk)  
    return h.hexdigest()  
  
def download_drunet_color(model_dir: str = DEFAULT_DIR) -> Path:  
    out_dir = Path(model_dir)  
    out_dir.mkdir(parents=True, exist_ok=True)  
    out_path = out_dir / FILENAME  
    
    if out_path.exists():  
        if KNOWN_SHA256:  
            if sha256sum(out_path) == KNOWN_SHA256:  
                print(f"Already present with matching hash: {out_path}")  
                return out_path  
            else:  
                print("Existing file hash mismatch, re-downloading...")  
                out_path.unlink()  
        else:  
            print(f"Already exists, skipping: {out_path}")  
            return out_path  
  
    #print(f"Downloading to {out_path} ...")  
    with requests.get(URL, stream=True, timeout=60) as r:  
        r.raise_for_status()  
        with open(out_path, "wb") as f:  
            for chunk in r.iter_content(chunk_size=1024 * 1024):  
                if chunk:  
                    f.write(chunk)  
  
    if KNOWN_SHA256:  
        got = sha256sum(out_path)  
        if got != KNOWN_SHA256:  
            out_path.unlink(missing_ok=True)  
            raise RuntimeError(f"SHA256 mismatch: got {got}")  
  
    #print("Done.")  
    return out_path  
  
if __name__ == "__main__": 
    p = download_drunet_color(DEFAULT_DIR)  
    print(f"Saved at: {p}")