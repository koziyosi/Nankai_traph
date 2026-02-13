import unittest
import numpy as np
import os
import sys

# Add root to path
sys.path.append(os.getcwd())

from src.physics.friction import RateStateFriction
from src.physics.stress_kernel import StressKernel

class TestRateStateFriction(unittest.TestCase):
    def setUp(self):
        self.friction = RateStateFriction()

    def test_friction_coefficient(self):
        # Basic test case
        V = np.array([1.0e-6])
        theta = np.array([1.0e-2]) # theta = L/V roughly (not exactly steady state for V0)
        a = np.array([0.005])
        b = np.array([0.008])
        L = np.array([0.4])

        mu = self.friction.compute_friction_coefficient(V, theta, a, b, L)

        # Check basic properties
        self.assertEqual(len(mu), 1)
        self.assertTrue(np.all(mu > 0))

        # Check steady state (V=V0, theta=L/V0)
        # Note: Implementation uses log(x + 1) for regularization
        # So at steady state, mu = mu0 + (a+b)*ln(2)
        theta_ss = np.array([L[0] / self.friction.V_0])
        V_ss = np.array([self.friction.V_0])

        mu_ss = self.friction.compute_friction_coefficient(V_ss, theta_ss, a, b, L)

        expected_mu = self.friction.mu_0 + (a[0] + b[0]) * np.log(2.0)
        self.assertAlmostEqual(mu_ss[0], expected_mu, places=5)

    def test_state_evolution(self):
        V = np.array([1.0e-9]) # Very slow
        theta = np.array([1.0])
        L = np.array([0.4])

        dtheta_dt = self.friction.compute_state_evolution(V, theta, L)
        self.assertEqual(len(dtheta_dt), 1)

class TestStressKernel(unittest.TestCase):
    def setUp(self):
        self.kernel = StressKernel(G=30.0e9, nu=0.25)

    def test_kernel_computation(self):
        # Create a simple 2-cell geometry
        centers = np.array([[0, 0, 10], [10, 0, 10]]) # 10km apart
        areas = np.array([1.0, 1.0]) # 1km^2
        normals = np.array([[0, 0, -1], [0, 0, -1]]) # Upward

        K = self.kernel.compute(centers, areas, normals)

        self.assertEqual(K.shape, (2, 2))

        # Self-stress should be negative (shear stress drops with slip)
        self.assertTrue(K[0, 0] < 0)
        self.assertTrue(K[1, 1] < 0)

        # Symmetry (approximate for far field, but exact for identical cells?)
        # For general slip, K_ij might not be symmetric, but let's check basic properties
        self.assertFalse(np.isnan(K).any())

if __name__ == '__main__':
    unittest.main()
