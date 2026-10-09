#!/usr/bin/env python3
"""
upload_to_hf.py
---------------
Uploads pre-processed Hugging Face Arrow datasets (data/hf_datasets/)
directly to the Hugging Face Hub as a private (or public) dataset repository.

Usage:
  # 1. Login to Hugging Face (requires a token with 'Write' permission):
  huggingface-cli login

  # 2. Upload Phase 1 datasets (combined, iemocap_session1, cafe):
  python upload_to_hf.py --repo_id <your-hf-username>/mtkd4ser-phase1

  # To upload all 5 IEMOCAP sessions:
  python upload_to_hf.py --repo_id <your-hf-username>/mtkd4ser-phase1 --all_sessions
"""

import os
import argparse
from datasets import load_from_disk
from huggingface_hub import HfApi, login


def upload_datasets(repo_id, hf_dir, all_sessions=False, private=True, token=None):
    if token:
        login(token=token)

    api = HfApi()
    user_info = api.whoami()
    print(f"Logged in to Hugging Face as: {user_info['name']}")

    # Create the dataset repository if it doesn't already exist
    print(f"Ensuring repository exists: {repo_id} (private={private})...")
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    # Core datasets needed for Phase 1
    configs_to_upload = [
        ("combined", os.path.join(hf_dir, "combined")),
        ("iemocap_session1", os.path.join(hf_dir, "iemocap_session1")),
        ("cafe", os.path.join(hf_dir, "cafe")),
    ]

    if all_sessions:
        for s in range(2, 6):
            s_name = f"iemocap_session{s}"
            s_path = os.path.join(hf_dir, s_name)
            if os.path.isdir(s_path):
                configs_to_upload.append((s_name, s_path))

    for config_name, path in configs_to_upload:
        if not os.path.isdir(path):
            print(f"[SKIP] Directory not found: {path}")
            continue

        print(f"\n" + "=" * 60)
        print(f"Loading '{config_name}' from: {path}...")
        ds = load_from_disk(path)
        print(f"Pushing config '{config_name}' to {repo_id}...")
        ds.push_to_hub(
            repo_id=repo_id,
            config_name=config_name,
            private=private,
        )
        print(f"Successfully uploaded config '{config_name}'!")

    print("\n" + "=" * 60)
    print(f"All datasets uploaded successfully to: https://huggingface.co/datasets/{repo_id}")
    print("=" * 60)


def main():
    _project_dir = os.path.dirname(os.path.abspath(__file__))
    _default_hf_dir = os.path.join(_project_dir, "data", "hf_datasets")

    parser = argparse.ArgumentParser(description="Upload SER datasets to Hugging Face Hub.")
    parser.add_argument("--repo_id", required=True,
                        help="Target repository on HF Hub, e.g. <username>/mtkd4ser-phase1")
    parser.add_argument("--hf_dir", default=_default_hf_dir,
                        help="Path to local data/hf_datasets directory")
    parser.add_argument("--token", default=None,
                        help="Optional Hugging Face user access token (write access)")
    parser.add_argument("--public", action="store_true", default=False,
                        help="Set dataset as public (Caution: IEMOCAP requires license, keep private!)")
    parser.add_argument("--all_sessions", action="store_true", default=False,
                        help="Also upload IEMOCAP sessions 2 through 5")
    args = parser.parse_args()

    upload_datasets(
        repo_id=args.repo_id,
        hf_dir=args.hf_dir,
        all_sessions=args.all_sessions,
        private=not args.public,
        token=args.token,
    )


if __name__ == "__main__":
    main()
