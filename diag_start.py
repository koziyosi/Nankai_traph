
print("Diag: Start")
import sys
print("Diag: Importing numpy")
import numpy as np
print("Diag: Importing cupy")
try:
    import cupy as cp
    print("Diag: CuPy imported")
    try:
        print("Diag: Checking Device(0)")
        d = cp.cuda.Device(0)
        print(f"Diag: Device name = {d.name}")
    except Exception as e:
        print(f"Diag: Device check failed: {e}")
except ImportError:
    print("Diag: CuPy import failed")

print("Diag: Importing src.solver.parallel")
try:
    from src.solver.parallel import get_system_info
    print("Diag: parallel imported")
    info = get_system_info()
    print(f"Diag: system info = {info}")
except Exception as e:
    print(f"Diag: parallel failed: {e}")

print("Diag: Importing main modules")
from src.geometry import PlateBoundary
print("Diag: geometry imported")
from src.physics import QuasiDynamicEquations
print("Diag: physics imported")

print("Diag: Done")
