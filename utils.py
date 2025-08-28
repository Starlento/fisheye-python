import numpy as np


def make_patch_grid(H: int, W: int, patch_size: int) -> np.ndarray:
    """
    Generate an RGB image of size (H, W) composed of patch_size x patch_size tiles.
    Adjacent tiles (up, down, left, right) are guaranteed different colors.

    Returns:
        np.ndarray of shape (H, W, 3), dtype=uint8.
    """
    if not (isinstance(H, int) and isinstance(W, int) and isinstance(patch_size, int)):
        raise TypeError("H, W, and patch_size must be integers.")
    if H <= 0 or W <= 0 or patch_size <= 0:
        raise ValueError("H, W, and patch_size must be positive.")

    # Number of tiles needed to fully cover the image
    rows = (H + patch_size - 1) // patch_size
    cols = (W + patch_size - 1) // patch_size

    # A small bright palette (k >= 2 ensures adjacent tiles differ with (i+j) % k)
    colors = np.array(
        [
            [239, 71, 111],  # pink/red
            [17, 138, 178],  # teal
            [6, 214, 160],  # green
            [255, 209, 102],  # yellow
        ],
        dtype=np.uint8,
    )
    k = colors.shape[0]

    # Checker-like pattern indices so neighbors differ
    idx = (np.add.outer(np.arange(rows), np.arange(cols))) % k  # (rows, cols)

    # Map indices to colors -> (rows, cols, 3)
    grid_colors = colors[idx]

    # Expand each tile to patch_size using repeat, then crop to exact HxW
    img = np.repeat(np.repeat(grid_colors, patch_size, axis=0), patch_size, axis=1)
    img = img[:H, :W, :]

    return img
