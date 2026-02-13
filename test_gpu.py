
import sys
import os

print("Testing CuPy import...")
try:
    import cupy as cp
    print(f"CuPy version: {cp.__version__}")
    
    print("Checking GPU device...")
    try:
        dev = cp.cuda.Device(0)
        print(f"Device 0: {dev}")
        props = cp.cuda.runtime.getDeviceProperties(0)
        print(f"  Name: {props['name'].decode('utf-8')}")
        print(f"  Compute Capability: {props['major']}.{props['minor']}")
        
        print("Testing array operation on GPU...")
        x_gpu = cp.array([1, 2, 3])
        print(f"  x_gpu: {x_gpu}")
        print(f"  x_gpu^2: {x_gpu ** 2}")
        print("CuPy test successful!")
        
    except Exception as e:
        print(f"GPU Device Error: {e}")

except ImportError as e:
    print(f"CuPy Import Error: {e}")
    print("Detailed info:")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()
