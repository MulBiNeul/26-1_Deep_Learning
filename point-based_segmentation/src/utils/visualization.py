from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def save_mask(mask: np.ndarray, save_path: str):
    """
    Save a binary segmentation mask as an image.

    Args:
        mask (np.ndarray): Binary mask (H, W) or with singleton dimensions
        save_path (str): Path to save the mask image
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    mask = np.squeeze(mask)

    if mask.ndim != 2:
        raise ValueError(f"Invalid mask shape for saving: {mask.shape}")

    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_prompt_image(image, points, labels, save_path: str):
    """
    Save an image with foreground/background points visualized.

    Args:
        image (PIL.Image): Input image
        points (list): List of point coordinates
        labels (list): List of labels (1=foreground, 0=background)
        save_path (str): Path to save the visualization
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    image_np = np.array(image)

    plt.figure(figsize=(8, 8))
    plt.imshow(image_np)

    for (x, y), label in zip(points, labels):
        if label == 1:
            plt.scatter(
                x, y,
                s=100,
                marker="o",
                edgecolors="red",
                facecolors="none",
                linewidths=2
            )
        else:
            plt.scatter(
                x, y,
                s=100,
                marker="x",
                c="blue",
                linewidths=2
            )

        plt.text(
            x + 8, y - 8,
            f"({int(x)},{int(y)})",
            color="black",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
        )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_overlay(image, mask: np.ndarray, save_path: str):
    """
    Save an overlay image with the segmentation mask applied.

    Args:
        image (PIL.Image): Input image
        mask (np.ndarray): Binary mask
        save_path (str): Path to save the overlay image
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    mask = np.squeeze(mask)

    if mask.ndim != 2:
        raise ValueError(f"Invalid mask shape for overlay: {mask.shape}")

    image_np = np.array(image)

    color_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
    color_mask[..., 0] = 1.0  # red channel
    color_mask[..., 3] = mask * 0.45  # alpha

    plt.figure(figsize=(8, 8))
    plt.imshow(image_np)
    plt.imshow(color_mask)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_three_panel_figure(original_image, prompt_image_path, overlay_image_path, save_path: str):
    """
    Save a combined visualization with three panels:
    original image, prompt points, and segmentation overlay.

    Args:
        original_image (PIL.Image): Original image
        prompt_image_path (str): Path to prompt visualization image
        overlay_image_path (str): Path to overlay image
        save_path (str): Path to save the combined figure
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_img = plt.imread(prompt_image_path)
    overlay_img = plt.imread(overlay_image_path)
    original_np = np.array(original_image)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(original_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(prompt_img)
    axes[1].set_title("Prompt Points")
    axes[1].axis("off")

    axes[2].imshow(overlay_img)
    axes[2].set_title("Segmentation Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()