import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates


# ---------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------

IMAGE_PATH = "image.png"

alpha = 0.01
beta = 0.30
gamma = 1.0

sigma = 2.0
edge_weight = 10.0

# Maximum number of iterations
max_iterations = 600

num_points = 120

# IMPROVEMENT 1:
# Maximum distance that one snake point can move
# during a single iteration.
max_px_move = 1.0

# IMPROVEMENT 2:
# Stop when average contour movement becomes very small.
convergence_tolerance = 0.01

# Avoid checking convergence immediately.
min_iterations = 20


# ---------------------------------------------------------
# 1. LOAD IMAGE
# ---------------------------------------------------------

image = Image.open(IMAGE_PATH).convert("L")
image = np.array(image, dtype=np.float64)

# Normalize image to [0, 1]
image = image / 255.0

height, width = image.shape


# ---------------------------------------------------------
# 2. SMOOTH IMAGE
# ---------------------------------------------------------

smooth_image = gaussian_filter(
    image,
    sigma=sigma
)


# ---------------------------------------------------------
# 3. IMAGE ENERGY
# ---------------------------------------------------------

Iy, Ix = np.gradient(smooth_image)

gradient_magnitude_squared = Ix**2 + Iy**2

# Normalize to make parameter tuning easier
gradient_magnitude_squared /= (
    gradient_magnitude_squared.max() + 1e-12
)

E_edge = -edge_weight * gradient_magnitude_squared


# Gradient of external energy
Fy, Fx = np.gradient(E_edge)


# ---------------------------------------------------------
# 4. INITIAL SNAKE
# ---------------------------------------------------------

center_x = 0.48 * width
center_y = 0.53 * height

radius_x = 0.45 * width
radius_y = 0.42 * height

theta = np.linspace(
    0,
    2 * np.pi,
    num_points,
    endpoint=False
)

x = center_x + radius_x * np.cos(theta)
y = center_y + radius_y * np.sin(theta)

initial_x = x.copy()
initial_y = y.copy()


# ---------------------------------------------------------
# 5. INTERNAL ENERGY MATRIX
# ---------------------------------------------------------

A = np.zeros((num_points, num_points))

for i in range(num_points):

    A[i, i] = 2 * alpha + 6 * beta

    A[i, (i - 1) % num_points] = -alpha - 4 * beta
    A[i, (i + 1) % num_points] = -alpha - 4 * beta

    A[i, (i - 2) % num_points] = beta
    A[i, (i + 2) % num_points] = beta


# Matrix used in the iterative solution
B = np.linalg.inv(
    A + gamma * np.eye(num_points)
)


# ---------------------------------------------------------
# 6. MOVE THE SNAKE
# ---------------------------------------------------------

for iteration in range(max_iterations):

    # Save current contour position so that movement
    # can be measured after the update.
    old_x = x.copy()
    old_y = y.copy()

    # Evaluate image force at every snake point
    fx = map_coordinates(
        Fx,
        [y, x],
        order=1,
        mode="nearest"
    )

    fy = map_coordinates(
        Fy,
        [y, x],
        order=1,
        mode="nearest"
    )

    # Calculate proposed new snake position
    proposed_x = B @ (gamma * x - fx)
    proposed_y = B @ (gamma * y - fy)


    # -----------------------------------------------------
    # IMPROVEMENT 1:
    # Limit the displacement of each point.
    # This prevents very large movements in one iteration.
    # -----------------------------------------------------

    dx = proposed_x - x
    dy = proposed_y - y

    displacement = np.sqrt(dx**2 + dy**2)

    # Scale movements that are larger than max_px_move
    scale = np.minimum(
        1.0,
        max_px_move / (displacement + 1e-12)
    )

    dx = dx * scale
    dy = dy * scale

    x = x + dx
    y = y + dy


    # Keep snake inside image
    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)


    # -----------------------------------------------------
    # IMPROVEMENT 2:
    # Measure how much the snake moved.
    # Stop automatically when it has converged.
    # -----------------------------------------------------

    movement = np.sqrt(
        (x - old_x)**2 +
        (y - old_y)**2
    )

    mean_movement = np.mean(movement)

    if (
        iteration >= min_iterations
        and mean_movement < convergence_tolerance
    ):
        print(
            f"Snake converged at iteration {iteration + 1}"
        )

        print(
            f"Mean movement: {mean_movement:.6f} pixels"
        )

        break

else:

    print(
        f"Maximum iterations reached: {max_iterations}"
    )


# ---------------------------------------------------------
# 7. DISPLAY RESULTS
# ---------------------------------------------------------

plt.figure(figsize=(8, 8))

plt.imshow(image, cmap="gray")

# Initial contour
plt.plot(
    np.append(initial_x, initial_x[0]),
    np.append(initial_y, initial_y[0]),
    "r--",
    linewidth=2,
    label="Initial snake"
)

# Final contour
plt.plot(
    np.append(x, x[0]),
    np.append(y, y[0]),
    "b-",
    linewidth=2,
    label="Final snake"
)

plt.legend()

plt.title(
    f"Snake Active Contour - "
    f"{iteration + 1} iterations"
)

plt.axis("off")

plt.show()