import sys
import os

# Add workspace root to sys.path so we can import the new module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cida_inference import CIDAWrapper, LatentMemory, LatentReasoner, LatentAgent

print("Checking cida_inference imports...")
print("CIDAWrapper:", CIDAWrapper)
print("LatentMemory:", LatentMemory)
print("LatentReasoner:", LatentReasoner)
print("LatentAgent:", LatentAgent)
print("Package structure verified successfully!")
