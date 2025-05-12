import math
import itertools
import numpy as np
import torch
from torchvision.transforms import GaussianBlur
import os
import torchvision.utils as vutils
import torch
import torch.nn as nn
import torch.optim as optim
import random

def save_images(epoch, client_id, batch_idx, X_transformed, X_perturbed):
    if epoch != 11 and epoch != 21 and epoch != 31 and epoch != 41:
        return  
    base_dir = "Transformed_Images_" + str(epoch)
    unperturbed_dir = os.path.join(base_dir, "Unperturbed")
    perturbed_dir = os.path.join(base_dir, "Perturbed")
    
    # Create directories if they don't exist
    os.makedirs(unperturbed_dir, exist_ok=True)
    os.makedirs(perturbed_dir, exist_ok=True)
    
    # Save the first 5 images from the batch
    for i in range(min(5, X_transformed.shape[0])):
        unperturbed_path = os.path.join(unperturbed_dir, f"client_{client_id}_batch_{batch_idx}_img_{i}.png")
        perturbed_path = os.path.join(perturbed_dir, f"client_{client_id}_batch_{batch_idx}_img_{i}.png")
        
        vutils.save_image(X_transformed[i], unperturbed_path)
        vutils.save_image(X_perturbed[i], perturbed_path)
        

def generate_permutations(grid_size, num_permutations=1000, save_path='./permutations.pt'):
    num_patches = grid_size * grid_size
    print(f"Number of patches {num_patches}")
    max_permutations = math.factorial(num_patches)

    if num_permutations >= max_permutations:
        permutations = list(itertools.permutations(range(num_patches)))
        permutations = [perm for perm in permutations if perm != tuple(range(num_patches))]
    else:
        permutations = set()
        while len(permutations) < num_permutations:
            perm = tuple(np.random.permutation(num_patches))
            if perm != tuple(range(num_patches)):
                permutations.add(perm)
        permutations = list(permutations)

    permutation_tensor = torch.tensor(permutations[:num_permutations])
    print(f"number of permutations {len(permutations)}")
    torch.save(permutation_tensor, save_path)
    print(f"Permutations saved to {save_path}")
    return len(permutations)

def generate_shuffled_tensors(tensor, grid_size=2, path = 'permutations.pt'):
    predefined_permutations = torch.load(path)
    batch_size, channels, height, width = tensor.shape
    patch_h, patch_w = height // grid_size, width // grid_size
    num_patches = grid_size * grid_size

    # Split image into patches
    patches = tensor.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(batch_size, num_patches, channels, patch_h, patch_w)

    # Choose a random permutation for each image in the batch
    perm_indices = torch.randint(0, len(predefined_permutations), (batch_size,))
    shuffled_images = []
    for i, perm_index in enumerate(perm_indices):
        perm = predefined_permutations[perm_index]
        shuffled_patches = patches[i][perm]  # Reorder patches based on the permutation
        shuffled_image = shuffled_patches.reshape(grid_size, grid_size, channels, patch_h, patch_w)
        shuffled_image = shuffled_image.permute(2, 0, 3, 1, 4).reshape(channels, height, width)
        shuffled_images.append(shuffled_image)

    # Stack shuffled images to form the batch and return perm_indices as target labels
    shuffled_images = torch.stack(shuffled_images)
    return shuffled_images, perm_indices

def smooth_patch_edges(images, grid_size=2, blur_kernel_size=5):
    batch_size, channels, height, width = images.shape
    patch_h, patch_w = height // grid_size, width // grid_size

    smoothed_images = images.clone()  # Copy to avoid modifying the original batch
    blur = GaussianBlur(kernel_size=(blur_kernel_size, blur_kernel_size))

    for b in range(batch_size):  # Iterate over batch
        # Apply smoothing to horizontal and vertical patch boundaries
        for i in range(1, grid_size):  # Ignore the outermost edges
            # Horizontal edges
            start_y = i * patch_h - blur_kernel_size // 2
            end_y = start_y + blur_kernel_size
            smoothed_images[b, :, start_y:end_y, :] = blur(smoothed_images[b, :, start_y:end_y, :])

            # Vertical edges
            start_x = i * patch_w - blur_kernel_size // 2
            end_x = start_x + blur_kernel_size
            smoothed_images[b, :, :, start_x:end_x] = blur(smoothed_images[b, :, :, start_x:end_x])

    return smoothed_images

