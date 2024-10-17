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

def generate_multiple_zernike_surfaces(num_surfaces, width, height, grid_size):
    """Generate multiple Zernike surfaces."""
    zernike_orders = [(random.randint(1, 8), random.randint(-n, n)) for n in range(1, num_surfaces + 1)]
    surfaces = [generate_zernike_surface_on_rectangular_grid(n, m, width, height, grid_size) for n, m in zernike_orders]
    return surfaces, zernike_orders

def randomly_combine_surfaces(orders, surfaces, num_combinations):
    """Randomly combine 4 to 6 surfaces from the given list."""
    combined_surfaces = []
    
    for _ in range(num_combinations):
        num_to_combine = random.randint(4, 6)  # Randomly choose to combine 4 to 6 surfaces
        selected_indices = random.sample(range(len(surfaces)), num_to_combine)  # Randomly select indices
        selected_surfaces = [surfaces[i] for i in selected_indices]  # Select surfaces based on indices
        combined_surface = combine_zernike_surfaces(selected_surfaces, width, height, grid_size)
        combined_surfaces.append(combined_surface)
    
    return combined_surfaces

def compare_surfaces(surface1, surface2):
    """Compare two surfaces using SSIM."""
    # Ensure surfaces are not flat and have some variation
    if np.all(surface1 == surface1[0]) or np.all(surface2 == surface2[0]):
        return 0  # If either surface is flat, return 0 similarity
    
    ssim_index, _ = ssim(surface1, surface2, full=True, data_range=surface1.max() - surface1.min())
    return ssim_index

def find_best_matching_sets(surfaces, num_surfaces_to_match=2, num_sets=5):
    """Find the best matching sets of surfaces based on SSIM."""
    all_indices = set(range(len(surfaces)))  # Set of all surface indices
    found_sets = []  # List to store found sets

    while len(found_sets) < num_sets and all_indices:
        best_set = None
        best_similarity = -1

        for indices in combinations(all_indices, num_surfaces_to_match):
            similarity_score = 0
            # Calculate the average SSIM for the selected surfaces
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    similarity_score += compare_surfaces(surfaces[indices[i]], surfaces[indices[j]])
            similarity_score /= (len(indices) * (len(indices) - 1)) / 2  # Average similarity score

            # Check if this set is the best found so far
            if similarity_score > best_similarity:
                best_similarity = similarity_score
                best_set = indices

        # If a best set is found, add it to found_sets and remove the indices from all_indices
        if best_set:
            found_sets.append((best_set, best_similarity))
            all_indices.difference_update(best_set)  # Remove used indices from available indices

    return found_sets  # Return found sets

def plot_surfaces(surfaces, best_sets):
    """Plot all surfaces in a grid and highlight the best matching sets."""
    num_surfaces = len(surfaces)
    cols = int(np.ceil(np.sqrt(num_surfaces)))  # Columns
    rows = int(np.ceil(num_surfaces / cols))   # Rows

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    for i, (surface, ax) in enumerate(zip(surfaces, axes)):
        ax.imshow(surface, cmap='viridis', extent=(-1, 1, -1, 1))
        ax.set_title(f'Surface {i + 1}')
        ax.axis('off')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Zernike Surface Height Maps', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for the title
    plt.show()

    # Plot the best matching sets side by side
    for indices, score in best_sets:
        fig, axes = plt.subplots(1, len(indices), figsize=(12, 6))
        
        for ax, idx in zip(axes, indices):
            ax.imshow(surfaces[idx], cmap='viridis', extent=(-1, 1, -1, 1))
            ax.set_title(f'Surface {idx + 1}')
            ax.axis('off')

        plt.suptitle(f'Best Matching Set - Similarity: {score:.2f}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

# Parameters for surface generation
num_surfaces = 10
width, height = 1.4, 1.4  # Width and height of the rectangular surface
grid_size = 20  # Resolution of the grid

# Step 1: Generate 60 individual Zernike surfaces
surfaces, zernike_orders = generate_multiple_zernike_surfaces(num_surfaces, width, height, grid_size)

# Step 2: Randomly combine surfaces to create a new set of surfaces
num_combinations = 20  # Number of new combined surfaces to create
combined_surfaces = randomly_combine_surfaces(zernike_orders, surfaces, num_combinations)

# Step 3: Find the best matching sets among the combined surfaces
num_surfaces_to_match = 5  # Define the number of surfaces to match
num_sets_to_find = 4  # Define the number of best matching sets to find
best_sets = find_best_matching_sets(combined_surfaces, num_surfaces_to_match, num_sets_to_find)

print ('Best sets found:', best_sets)

# Step 4: Plot the combined surfaces and highlight the best matches
plot_surfaces(combined_surfaces, best_sets)