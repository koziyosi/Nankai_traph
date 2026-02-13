
import numpy as np
import sys

print("Loading CuPy...")
try:
    import cupy as cp
    from src.physics.cupy_kernels import get_velocity_solver_kernel
    
    print("Creating Kernel...")
    kernel = get_velocity_solver_kernel()
    
    print("Preparing Data...")
    n = 10
    tau = cp.ones(n, dtype=cp.float64) * 1e6
    theta = cp.ones(n, dtype=cp.float64)
    a = cp.ones(n, dtype=cp.float64) * 0.01
    b = cp.ones(n, dtype=cp.float64) * 0.02
    L = cp.ones(n, dtype=cp.float64) * 0.02
    sigma_eff = cp.ones(n, dtype=cp.float64) * 1e7
    G = 30e9
    beta = 3750.0
    eta = 1.0
    mu_0 = 0.6
    V_0 = 1e-6
    V_c = 1e-8
    
    print("Running Kernel (Compilation triggers here)...")
    try:
        V = kernel(tau, theta, a, b, L, sigma_eff, G, beta, eta, mu_0, V_0, V_c)
        print("Kernel execution successful!")
        print(f"V[0] = {V[0]}")
    except Exception as e:
        print(f"Kernel execution FAILED: {e}")
        import traceback
        traceback.print_exc()

except ImportError:
    print("CuPy import failed")
except Exception as e:
    print(f"Setup failed: {e}")
