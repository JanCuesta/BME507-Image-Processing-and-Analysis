# Improved implementation of Otsu's method

from PIL import Image
from torchvision import transforms
import torch
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# READ IMAGE
# ------------------------------------------------------------

img = Image.open("./Chest_Xray.png")

# Convert image to grayscale
img = img.convert("L")

# Convert image to tensor [0, 1]
to_tensor = transforms.ToTensor()
img_tensor = to_tensor(img)

print("Image shape:", img_tensor.shape)


# ------------------------------------------------------------
# FUNCTION TO CALCULATE OTSU THRESHOLD
# ------------------------------------------------------------

def otsu_threshold(image_tensor):

    # Flatten image
    pixel_values = image_tensor.flatten()

    # Convert pixel values from [0, 1] to [0, 255]
    pixel_values_255 = (pixel_values * 255).byte()

    # Count pixels at each intensity level
    pixel_count = torch.zeros(256)

    for i in range(256):
        pixel_count[i] = (pixel_values_255 == i).sum().item()

    # Total number of pixels
    N = len(pixel_values_255)

    # Probability of each intensity level
    p_i = pixel_count / N

    # Intensity levels: 0, 1, ..., 255
    intensity_levels = torch.arange(256, dtype=torch.float32)

    # Total mean of the image
    mu_total = torch.sum(intensity_levels * p_i)

    # Total variance
    sigma_total_squared = torch.sum(
        ((intensity_levels - mu_total) ** 2) * p_i
    )

    # --------------------------------------------------------
    # IMPROVEMENT 1:
    # Precompute cumulative probability and cumulative mean.
    # This avoids recalculating the same sums for every k.
    # --------------------------------------------------------

    cumulative_probability = torch.cumsum(p_i, dim=0)

    cumulative_mean = torch.cumsum(
        intensity_levels * p_i,
        dim=0
    )

    best_DC3 = -1
    optimal_threshold = 0

    # Test every possible threshold
    for k in range(255):

        # Class probabilities using cumulative values
        w_0 = cumulative_probability[k]
        w_1 = 1 - w_0

        # ----------------------------------------------------
        # IMPROVEMENT 2:
        # Skip thresholds that create an empty class.
        # This prevents division by zero.
        # ----------------------------------------------------

        if w_0 <= 1e-12 or w_1 <= 1e-12:
            continue

        # Mean of class C0
        mu_0 = cumulative_mean[k] / w_0

        # Mean of class C1
        mu_1 = (
            mu_total - cumulative_mean[k]
        ) / w_1

        # Between-class variance
        sigma_b_squared = (
            w_0 * (mu_0 - mu_total) ** 2
            +
            w_1 * (mu_1 - mu_total) ** 2
        )

        # Discriminant criterion
        DC3 = sigma_b_squared / sigma_total_squared

        if DC3 > best_DC3:
            best_DC3 = DC3
            optimal_threshold = k

    return optimal_threshold, best_DC3


# ------------------------------------------------------------
# OTSU ON ORIGINAL IMAGE
# ------------------------------------------------------------

threshold_original, DC3_original = otsu_threshold(img_tensor)

print("\nOriginal image")
print("Optimal threshold:", threshold_original)
print("Maximum discriminant criterion:", DC3_original.item())


# ------------------------------------------------------------
# IMPROVEMENT 3:
# Apply Gaussian smoothing before Otsu thresholding.
# This allows comparison with the original implementation.
# ------------------------------------------------------------

gaussian_blur = transforms.GaussianBlur(
    kernel_size=5,
    sigma=1.0
)

smoothed_tensor = gaussian_blur(img_tensor)

threshold_smoothed, DC3_smoothed = otsu_threshold(
    smoothed_tensor
)

print("\nGaussian-smoothed image")
print("Optimal threshold:", threshold_smoothed)
print("Maximum discriminant criterion:", DC3_smoothed.item())


# ------------------------------------------------------------
# APPLY THRESHOLDS
# ------------------------------------------------------------

# FIX:
# Threshold is in [0,255], while tensor values are in [0,1].
# Therefore the correct conversion uses 255, not 225.

binary_original = (
    img_tensor >= threshold_original / 255.0
).float()

binary_smoothed = (
    smoothed_tensor >= threshold_smoothed / 255.0
).float()


# ------------------------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------------------------

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(img_tensor.squeeze(), cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(binary_original.squeeze(), cmap="gray")
plt.title(
    f"Otsu - Original\nThreshold = {threshold_original}"
)
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(smoothed_tensor.squeeze(), cmap="gray")
plt.title("Gaussian-Smoothed Image")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(binary_smoothed.squeeze(), cmap="gray")
plt.title(
    f"Otsu - Smoothed\nThreshold = {threshold_smoothed}"
)
plt.axis("off")

plt.tight_layout()
plt.show()