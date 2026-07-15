# Script to verify synchronization between cida_plugin/inference/ and cida_inference/
import os
import filecmp

def check_sync():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dir1 = os.path.join(base_dir, "cida_plugin", "inference")
    dir2 = os.path.join(base_dir, "cida_inference")
    
    print(f"Comparing:")
    print(f"  Dir 1: {dir1}")
    print(f"  Dir 2: {dir2}")
    
    if not os.path.exists(dir1) or not os.path.exists(dir2):
        print("[FAIL] Error: One of the directories does not exist.")
        return False

    # Get relative paths of all files in dir1
    files1 = []
    for root, _, filenames in os.walk(dir1):
        for f in filenames:
            if f.endswith('.py') and not f.startswith('__pycache__'):
                rel_path = os.path.relpath(os.path.join(root, f), dir1)
                files1.append(rel_path)

    # Get relative paths of all files in dir2
    files2 = []
    for root, _, filenames in os.walk(dir2):
        for f in filenames:
            if f.endswith('.py') and not f.startswith('__pycache__'):
                rel_path = os.path.relpath(os.path.join(root, f), dir2)
                files2.append(rel_path)

    mismatch = False
    
    # Check if any files in dir1 are missing in dir2
    for f in files1:
        f2_path = os.path.join(dir2, f)
        if not os.path.exists(f2_path):
            print(f"[FAIL] Mismatch: {f} exists in cida_plugin/inference but is missing in cida_inference")
            mismatch = True
        else:
            f1_path = os.path.join(dir1, f)
            if not filecmp.cmp(f1_path, f2_path, shallow=False):
                print(f"[FAIL] Content mismatch: {f} differs between cida_plugin/inference and cida_inference")
                mismatch = True

    # Check if any files in dir2 are missing in dir1
    for f in files2:
        f1_path = os.path.join(dir1, f)
        if not os.path.exists(f1_path):
            print(f"[FAIL] Mismatch: {f} exists in cida_inference but is missing in cida_plugin/inference")
            mismatch = True

    if mismatch:
        print("\n[FAIL] Verification FAILED: The packages are out of sync!")
        return False
    else:
        print("\n[SUCCESS] Verification PASSED: All files in cida_plugin/inference and cida_inference are identical and in sync.")
        return True

if __name__ == "__main__":
    import sys
    success = check_sync()
    sys.exit(0 if success else 1)
