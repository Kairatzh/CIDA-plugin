import os
import shutil

def clean_weights():
    paths_to_remove = [
        "cida_v5_rl_weights",
        "/content/drive/MyDrive/cida_v5_rl_weights",
        "C:\\content\\drive\\MyDrive\\cida_v5_rl_weights"
    ]
    
    removed_any = False
    for path in paths_to_remove:
        if os.path.exists(path):
            print(f"Found weights directory at: {path}")
            try:
                shutil.rmtree(path)
                print(f"✅ Successfully deleted: {path}")
                removed_any = True
            except Exception as e:
                print(f"❌ Failed to delete {path}: {e}")
                
    if not removed_any:
        print("ℹ️ No old CIDA v5 RL weights found to delete. You are good to go!")

if __name__ == "__main__":
    clean_weights()
