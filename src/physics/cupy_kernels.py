
"""
CuPy Kernels for Nankai Trough Simulation
GPU accelerated physics calculations
"""

try:
    import cupy as cp
except ImportError:
    cp = None

class VectorizedNewtonSolver:
    """
    Vectorized Newton solver using pure CuPy array operations.
    Avoids need for nvcc/compilation at runtime.
    """
    def __init__(self):
        pass
        
    def __call__(self, tau, theta, a, b, L, sigma_eff, G, beta, eta, mu_0, V_0, V_c):
        """
        Solve equilibrium velocity:
        F(V) = tau - sigma_eff * mu(V, theta) - eta*G/beta * V = 0
        """
        rad_coef = eta * G / beta
        
        # Initial guess
        V = cp.full_like(tau, V_0)
        theta_i = cp.maximum(theta, 1.0e-20)
        
        # Newton loop (fixed iterations for GPU efficiency avoiding divergent paths)
        max_iter = 50
        tol = 1.0e-10
        
        for _ in range(max_iter):
            V_safe = cp.maximum(V, 1.0e-20)
            
            # F(V)
            term1 = a * cp.log(V_safe / V_0 + 1.0)
            term2 = b * cp.log(V_0 * theta_i / L + 1.0)
            mu_v = mu_0 + term1 + term2
            
            F = tau - sigma_eff * mu_v - rad_coef * V_safe
            
            # F'(V)
            dmu_dV = a / (V_safe + V_0)
            dF_dV = -sigma_eff * dmu_dV - rad_coef
            
            # Update
            # Avoid division by zero (though dF_dV should be negative definitive)
            dV = -F / dF_dV
            
            V_new = V + dV
            
            # Limits
            V_new = cp.maximum(V_new, 1.0e-20)
            V_new = cp.minimum(V_new, 1.0e3)
            
            # Check convergence (optional optimization: stop if all converged)
            # But CPU-GPU sync is slow, so just checking every 10 steps or not at all is faster.
            # Here we just run fixed iterations or check relative change.
            # Using fixed iterations is often faster on GPU to avoid sync.
            
            V = V_new
            
            # Simple early exit check every 10 steps? 
            # if _ % 10 == 0 and cp.max(cp.abs(dV)) < tol: break
            
        return V

class VectorizedStateEvolution:
    def __call__(self, V, theta, L, V_c):
        """
        dtheta/dt = exp(-V*theta/L) - (V*theta/L)*exp(-Vc/V)
        """
        V_safe = cp.maximum(V, 1.0e-20)
        theta_safe = cp.maximum(theta, 1.0e-20)
        
        x = V_safe * theta_safe / L
        
        term1 = cp.zeros_like(x)
        mask1 = x < 700.0
        term1[mask1] = cp.exp(-x[mask1])
        
        term2 = cp.zeros_like(x)
        ratio = V_c / V_safe
        mask2 = ratio < 700.0
        term2[mask2] = x[mask2] * cp.exp(-ratio[mask2])
        
        return term1 - term2

def get_velocity_solver_kernel():
    if cp is None: return None
    return VectorizedNewtonSolver()

def get_state_evolution_kernel():
    if cp is None: return None
    return VectorizedStateEvolution()
