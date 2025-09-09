from huggingface_hub import snapshot_download
from pathlib import Path
import zipfile

dest = Path(r"C:\Users\William\Desktop\Projects\framegen\data")
dest.mkdir(parents=True, exist_ok=True)

print("Downloading UCF101 ZIP snapshot…")
repo_id = "quchenyuan/UCF101-ZIP"
repo_dir = snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=str(dest),
    local_dir_use_symlinks=False
)

print("Searching for ZIP files...")
zips = list(Path(repo_dir).rglob("*.zip"))
if not zips:
    print("No ZIP file found—download might have failed!")
for z in zips:
    print(f"Unzipping {z.name} ...")
    with zipfile.ZipFile(z, "r") as zf:
        zf.extractall(dest)

print("Done! Dataset extracted to:", dest)
