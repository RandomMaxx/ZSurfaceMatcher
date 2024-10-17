# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 22:50:52 2024

@author: Frank
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
import random
from skimage.metrics import structural_similarity as ssim
from itertools import combinations

def zernike_radial(n, m, rho):
    """Calculate the Zernike radial polynomial."""
    R = np.zeros_like(rho)
    for k in range((n - abs(m)) // 2 + 1):
        coeff = ((-1) ** k) * factorial(n - k) / (
            factorial(k) * factorial((n + abs(m)) // 2 - k) * factorial((n - abs(m)) // 2 - k)
        )
        R += coeff * rho ** (n - 2 * k)
    return R

def zernike(n, m, rho, theta):
    """Calculate the Zernike polynomial."""
    if m >= 0:
        return zernike_radial(n, m, rho) * np.cos(m * theta)
    else:
        return zernike_radial(n, -m, rho) * np.sin(-m * theta)

def generate_zernike_surface_on_rectangular_grid(n, m, width, height, grid_size=100, coeff=1):
    """Generate a surface height map based on Zernike polynomial over a rectangular grid."""
    # Create a grid of points in Cartesian coordinates
    y, x = np.linspace(-height / 2, height / 2, grid_size), np.linspace(-width / 2, width / 2, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Convert Cartesian coordinates to polar coordinates
    rho = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    
    # Calculate the Zernike polynomial over the rectangular grid
    Z = np.zeros_like(rho)
    mask = rho <= 1  # Mask for points within the unit disk
    Z[mask] = coeff * zernike(n, m, rho[mask], theta[mask])  # Multiply by coefficient

    return Z

def combine_zernike_surfaces(surfaces, width, height, grid_size=100):
    """Combine multiple Zernike surfaces to create a more complex height map."""
    total_surface = np.zeros((grid_size, grid_size))
    for surface in surfaces:
        total_surface += surface
    return total_surface

def generate_multiple_zernike_surface_pairs(num_pairs, width, height, grid_size):
    """Generate pairs of Zernike surfaces (front and back)."""
    surfaces = []
    for _ in range(num_pairs):
        n, m = random.randint(1, 8), random.randint(-8, 8)
        front_surface = generate_zernike_surface_on_rectangular_grid(n, m, width, height, grid_size)
        back_surface = generate_zernike_surface_on_rectangular_grid(n, m, width, height, grid_size, coeff=-1)
        surfaces.append((front_surface, back_surface))
    return surfaces

def compare_surfaces(surface1, surface2):
    """Compare two surfaces using SSIM."""
    # Ensure surfaces are not flat and have some variation
    if np.all(surface1 == surface1[0]) or np.all(surface2 == surface2[0]):
        return 0  # If either surface is flat, return 0 similarity
    
    ssim_index, _ = ssim(surface1, surface2, full=True, data_range=surface1.max() - surface1.min())
    return ssim_index

def find_best_matching_surface_pairs(surface_pairs, num_sets=5):
    """Find the best matching pairs of front and back surfaces."""
    found_sets = []

    for _ in range(num_sets):
        best_set = None
        best_similarity = -1

        for i, (front1, back1) in enumerate(surface_pairs):
            for j, (front2, back2) in enumerate(surface_pairs):
                if i != j:  # Ensure we don't compare the same pair
                    similarity = compare_surfaces(back1, front2)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_set = (i, j)

        if best_set:
            found_sets.append((best_set, best_similarity))

    return found_sets

def plot_surface_pairs(surfaces, best_sets):
    """Plot best matching pairs of surfaces."""
    for (i, j), score in best_sets:
        front1, back1 = surfaces[i]
        front2, back2 = surfaces[j]
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        axes[0].imshow(front1, cmap='viridis', extent=(-1, 1, -1, 1))
        axes[0].set_title(f'Front Surface Pair {i + 1}')
        axes[0].axis('off')

        axes[1].imshow(back1, cmap='viridis', extent=(-1, 1, -1, 1))
        axes[1].set_title(f'Back Surface Pair {i + 1}')
        axes[1].axis('off')

        axes[2].imshow(front2, cmap='viridis', extent=(-1, 1, -1, 1))
        axes[2].set_title(f'Front Surface Pair {j + 1}')
        axes[2].axis('off')

        axes[3].imshow(back2, cmap='viridis', extent=(-1, 1, -1, 1))
        axes[3].set_title(f'Back Surface Pair {j + 1}')
        axes[3].axis('off')

        plt.suptitle(f'Best Match - Similarity: {score:.2f}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

# Parameters for surface generation
num_pairs = 30  # Number of front-back pairs
width, height = 1.4, 1.4  # Width and height of the rectangular surface
grid_size = 100  # Resolution of the grid

# Step 1: Generate pairs of Zernike surfaces (front and back)
surface_pairs = generate_multiple_zernike_surface_pairs(num_pairs, width, height, grid_size)

# Step 2: Find the best matching surface pairs
num_sets_to_find = 5  # Define the number of best matching sets to find
best_sets = find_best_matching_surface_pairs(surface_pairs, num_sets_to_find)

# Step 3: Plot the best matching surface pairs
plot_surface_pairs(surface_pairs, best_sets)