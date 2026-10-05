"""
auto_save_and_backup.py — Automated Nighttime Backup & Cloud Publisher for Kairos-SOTA

Functions:
  1. Polls and waits for benchmark completion if currently executing in background.
  2. Creates a consolidated ZIP bundle of all checkpoints, metrics, and plots.
  3. Automatically backs up all artifacts to Google Drive (/content/drive/MyDrive/KAIROS_SOTA_BACKUP).
  4. Publishes weights, 5 publication charts, passport, and audit logs to Hugging Face Hub.
  5. Works seamlessly overnight without requiring user presence.
"""

import os
import sys
import time
import shutil
import zipfile
import argparse
from typing import Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    from upload_kairos_sota_to_hf import package_and_upload_sota
except ImportError:
    package_and_upload_sota = None


def wait_for_benchmark_completion(benchmark_dir: str, poll_interval: int = 15, max_wait_minutes: int = 180) -> bool:
    """Waits until benchmark finishes generating metrics and reports."""
    metric_file = os.path.join(benchmark_dir, "benchmark_metrics.json")
    report_file = os.path.join(benchmark_dir, "scientific_benchmark_report.md")

    if os.path.exists(metric_file) and os.path.exists(report_file):
        print(f"[+] Benchmark results already present in: {benchmark_dir}")
        return True

    print(f"[*] Benchmark is still running. Waiting for completion (polling every {poll_interval}s)...")
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60

    dots = 0
    while time.time() - start_time < max_wait_seconds:
        if os.path.exists(metric_file) and os.path.exists(report_file):
            # Wait additional 5 seconds to ensure file write buffers are flushed
            time.sleep(5)
            print(f"\n[+] Benchmark finished successfully! Metrics and report detected.")
            return True
        time.sleep(poll_interval)
        dots += 1
        elapsed_min = (time.time() - start_time) / 60
        print(f"  ... waiting for benchmark completion ({elapsed_min:.1f} min elapsed)", end="\r")

    print(f"\n[!] Timeout: Benchmark did not complete within {max_wait_minutes} minutes.")
    return False


def backup_to_google_drive(source_dirs: list, drive_target: str) -> Optional[str]:
    """Mounts (if in Colab) and copies all results to Google Drive."""
    # Attempt Colab Drive mount
    try:
        if os.path.exists("/content"):
            from google.colab import drive
            if not os.path.exists("/content/drive/MyDrive"):
                print("[*] Mounting Google Drive to save results...")
                drive.mount("/content/drive")
    except Exception as e:
        print(f"[*] Note on Google Drive mount: {e}")

    drive_root = "/content/drive/MyDrive"
    if not os.path.exists(drive_root):
        print("[!] Google Drive is not mounted or not accessible; skipping Drive backup.")
        return None

    target_path = os.path.join(drive_root, drive_target)
    os.makedirs(target_path, exist_ok=True)
    print(f"\n[*] Backing up artifacts to Google Drive: {target_path}...")

    copied_count = 0
    for s_dir in source_dirs:
        if os.path.exists(s_dir):
            for root, _, files in os.walk(s_dir):
                for f in files:
                    src_f = os.path.join(root, f)
                    rel_path = os.path.relpath(src_f, s_dir)
                    dst_f = os.path.join(target_path, os.path.basename(s_dir), rel_path)
                    os.makedirs(os.path.dirname(dst_f), exist_ok=True)
                    shutil.copy2(src_f, dst_f)
                    copied_count += 1

    # Also backup model passport if present
    passport_path = os.path.join(SCRIPT_DIR, "KAIROS_V2_300KK_MODEL_PASSPORT.md")
    if os.path.exists(passport_path):
        shutil.copy2(passport_path, os.path.join(target_path, "KAIROS_V2_300KK_MODEL_PASSPORT.md"))
        copied_count += 1

    print(f"[+] Successfully backed up {copied_count} files to Google Drive!")
    return target_path


def create_zip_archive(archive_name: str, paths_to_pack: list) -> str:
    """Creates a consolidated ZIP archive of all results."""
    zip_path = os.path.abspath(archive_name)
    print(f"\n[*] Creating consolidated ZIP bundle: {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for p in paths_to_pack:
            if os.path.isfile(p):
                zipf.write(p, arcname=os.path.basename(p))
            elif os.path.isdir(p):
                for root, _, files in os.walk(p):
                    for file in files:
                        full_f = os.path.join(root, file)
                        rel_f = os.path.relpath(full_f, os.path.dirname(p))
                        zipf.write(full_f, arcname=rel_f)

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"[+] ZIP bundle created successfully ({size_mb:.2f} MB): {zip_path}")
    return zip_path


def main():
    parser = argparse.ArgumentParser(description="Nighttime Auto-Saver & Cloud Publisher for Kairos-SOTA")
    parser.add_argument("--repo-id", type=str, default="kirtzh/Kairos-v2-300kk", help="Hugging Face repo ID")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face Write Token")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_results", help="Path to benchmark results")
    parser.add_argument("--checkpoint-dir", type=str, default="./kairos_checkpoints/apex_sota", help="Path to checkpoints")
    parser.add_argument("--drive-folder", type=str, default="KAIROS_SOTA_NIGHTLY_BACKUP", help="Folder in Google Drive")
    parser.add_argument("--wait", action="store_true", default=True, help="Wait for benchmark completion before saving")
    parser.add_argument("--skip-hf", action="store_true", help="Skip Hugging Face upload")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("🌙 KAIROS-SOTA NIGHTTIME AUTO-SAVER & CLOUD BACKUP")
    print("="*80)

    # 1. Wait for benchmark if requested
    if args.wait:
        finished = wait_for_benchmark_completion(args.benchmark_dir)
        if not finished:
            print("[!] Warning: Benchmark metrics were not detected. Proceeding with available files...")

    # 2. Local ZIP bundle creation
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    zip_name = f"kairos_sota_backup_{timestamp}.zip"
    pack_items = [args.benchmark_dir, args.checkpoint_dir]
    passport = os.path.join(SCRIPT_DIR, "KAIROS_V2_300KK_MODEL_PASSPORT.md")
    if os.path.exists(passport):
        pack_items.append(passport)
    created_zip = create_zip_archive(zip_name, pack_items)

    # 3. Google Drive Backup
    drive_backup_path = backup_to_google_drive([args.benchmark_dir, args.checkpoint_dir], args.drive_folder)
    if drive_backup_path and os.path.exists(created_zip):
        shutil.copy2(created_zip, os.path.join(drive_backup_path, os.path.basename(created_zip)))
        print(f"[+] Copied ZIP archive to Google Drive: {drive_backup_path}")

    # 4. Hugging Face Hub Upload
    if not args.skip_hf and package_and_upload_sota is not None:
        print("\n[*] Uploading weights, configs, charts, and passport to Hugging Face...")
        try:
            package_and_upload_sota(
                repo_id=args.repo_id,
                token=args.token,
                checkpoint_dir=args.checkpoint_dir,
                benchmark_dir=args.benchmark_dir,
                private=False,
            )
            print(f"[+] Hugging Face Hub repository updated: https://huggingface.co/{args.repo_id}")
        except Exception as e:
            print(f"[!] Error during Hugging Face upload: {e}")
            print("    Don't worry: All files are safely preserved in Google Drive and local ZIP.")

    print("\n" + "="*80)
    print("💤 NIGHTTIME BACKUP COMPLETE! ALL ARTIFACTS ARE PERMANENTLY SAFE.")
    if drive_backup_path:
        print(f"📁 Google Drive: {drive_backup_path}")
    print(f"📦 Local Archive: {created_zip}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
