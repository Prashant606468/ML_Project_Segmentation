"""Upload / download the trained checkpoints (runs/<run>/best.pt) via the Hugging Face Hub.

The 18 checkpoints total ~1.4 GB (56-119 MB each), above GitHub's 100 MB file
limit, so they are hosted on the Hub. Integrity is checked against checkpoints.sha256.

    python scripts/checkpoints.py download --repo <user>/polyp-yolo-diff
    python scripts/checkpoints.py download --repo <user>/polyp-yolo-diff --runs yolo_seg_v8s_bnd_s0 pranet_s0
    python scripts/checkpoints.py upload   --repo <user>/polyp-yolo-diff     (maintainers; needs `huggingface-cli login`)
"""
import argparse
import hashlib
import os
import shutil

from huggingface_hub import HfApi, hf_hub_download

ROOT = os.path.join(os.path.dirname(__file__), "..")


def expected_hashes():
    out = {}
    for line in open(os.path.join(ROOT, "checkpoints.sha256")):
        digest, path = line.split()
        out[path.split("/")[1]] = digest
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def upload(repo, runs):
    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True)
    for run in runs:
        api.upload_file(path_or_fileobj=os.path.join(ROOT, "runs", run, "best.pt"),
                        path_in_repo=f"{run}/best.pt", repo_id=repo)
        print("uploaded", run)
    api.upload_file(path_or_fileobj=os.path.join(ROOT, "checkpoints.sha256"),
                    path_in_repo="checkpoints.sha256", repo_id=repo)


def download(repo, runs, hashes):
    for run in runs:
        dst = os.path.join(ROOT, "runs", run, "best.pt")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(hf_hub_download(repo, f"{run}/best.pt"), dst)
        ok = sha256(dst) == hashes[run]
        print(f"{run}: {'sha256 OK' if ok else 'SHA256 MISMATCH'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["upload", "download"])
    p.add_argument("--repo", required=True, help="Hugging Face model repo id, e.g. user/polyp-yolo-diff")
    p.add_argument("--runs", nargs="+", help="subset of run names (default: all 18)")
    a = p.parse_args()
    hashes = expected_hashes()
    runs = a.runs or sorted(hashes)
    if a.action == "upload":
        upload(a.repo, runs)
    else:
        download(a.repo, runs, hashes)


if __name__ == "__main__":
    main()
