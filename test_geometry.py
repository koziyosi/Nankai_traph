
import sys
import os
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

# Add current directory to path
sys.path.append(os.getcwd())

from src.geometry.plate_boundary import PlateBoundary

def test_plate_boundary():
    print("Testing PlateBoundary...")
    try:
        pb = PlateBoundary()
        print("PlateBoundary initialized.")
        
        if pb.depth_data_points is not None:
             print(f"Loaded {len(pb.depth_data_points)} data points.")
             print(f"Sample data: {pb.depth_data_points[:5]}")
        else:
             print("Data points not loaded.")
             return

        # Test interpolation
        test_lon = 135.0
        test_lat = 33.0
        depth = pb.get_depth(np.array([test_lon]), np.array([test_lat]))
        print(f"Depth at ({test_lon}, {test_lat}): {depth[0]} km")
        
        # Test grid generation
        print("Testing grid generation...")
        lon_grid, lat_grid, depth_grid = pb.get_plate_surface_grid(nx=10, ny=10)
        print(f"Grid shape: {depth_grid.shape}")
        print(f"Grid min/max depth: {np.nanmin(depth_grid):.2f} / {np.nanmax(depth_grid):.2f} km")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_plate_boundary()
