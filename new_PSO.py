import os
import random
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


image_original = Image.open("lung2.jpg").convert("RGB")
width, height = image_original.size
print(f"Image size:{width}w, {height}h")
rgb_values_original = np.array(image_original.getdata())

# bounding options
BOUND_VELOCITY = "clip"  # "clip", "reflection", None
BOUND_POSITION = "clip"  # "clip", "reflection"


def evaluate_palette_fitness(palette):
    """Fitness score = squared Euclidean distance between pixels and palette"""
    distances = cdist(rgb_values_original, palette, metric='sqeuclidean')
    min_distances = np.min(distances, axis=1)
    return np.sum(min_distances)


def clip_ro_rgb(values):
    """make values to be in range [0,255]"""
    return np.clip(values, 0, 255)


def apply_bounding(values, bound_type="clip", lower=0, upper=255):
    """
    bounding function to handle different strategies
    values: array (positions or velocities)
    bound_type: "clip", "reflection"
    lower, upper: min and max bounds
    """
    if bound_type == "clip":
        return np.clip(values, lower, upper)

    elif bound_type == "reflection":
        values = np.where(values > upper, 2 * upper - values, values)  # reflect if > max
        values = np.where(values < lower, 2 * lower - values, values)  # reflect if < min
        return np.clip(values, lower, upper)  # if still out of bounds after reflection, we do a clip


def visualize_and_save_palette(palette, iteration):
    """Display and save palette"""
    fig, ax = plt.subplots(figsize=(6, 2))

    for i, color in enumerate(palette):
        rect = plt.Rectangle((i, 0), 1, 1, color=color / 255)
        ax.add_patch(rect)
        ax.text(i + 0.5, -0.4, f'R: {int(color[0])}', ha='center', fontsize=10)
        ax.text(i + 0.5, -0.6, f'G: {int(color[1])}', ha='center', fontsize=10)
        ax.text(i + 0.5, -0.8, f'B: {int(color[2])}', ha='center', fontsize=10)

    ax.set_xlim(0, len(palette))
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    plt.title(f"Global Best Palette (Iteration {iteration+1})")
    plt.savefig(f"results/lung/palette_iter_{iteration+1}.png", bbox_inches='tight', dpi=150)
    plt.show()


def cluster_recreate_image_with_palette(rgb_values_original, palette, width, height, iteration, method="PSO"):
    """
    Assigns each pixel to the closest color in the palette, reconstructs the image,
    and appends the best palette with RGB labels below it before saving.
    """

    # compute Euclidean distance between each pixel and each color in the palette
    distances = cdist(rgb_values_original, palette, metric='sqeuclidean')

    # for each pixel, find the index of the closest color in the palette
    closest_palette_indices = np.argmin(distances, axis=1)

    # replace each pixel with the corresponding palette color
    clustered_image = palette[closest_palette_indices]

    clustered_image = clustered_image.reshape((height, width, 3)).astype(np.uint8)
    clustered_img = Image.fromarray(clustered_image)

    # visualisation settings
    palette_height = 50
    palette_width = width
    spacing = 0
    text_height = 40
    total_height = height + palette_height + spacing + text_height

    combined_img = Image.new("RGB", (width, total_height), (255, 255, 255))
    combined_img.paste(clustered_img, (0, 0))

    # palette visualization
    palette_image = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)
    num_colors = len(palette)
    segment_width = palette_width // num_colors

    for i, color in enumerate(palette):
        start_x = i * segment_width
        end_x = (i + 1) * segment_width if i < num_colors - 1 else palette_width
        palette_image[:, start_x:end_x] = color

    # convert palette to PIL and put it below the reconstructed image
    palette_img = Image.fromarray(palette_image)
    combined_img.paste(palette_img, (0, height + spacing))

    # write RGB values under the palette
    draw = ImageDraw.Draw(combined_img)
    # font_size = 8
    font = ImageFont.truetype("arial.ttf", 8)

    text_y = height + spacing + palette_height + 5

    for i, color in enumerate(palette):
        start_x = i * segment_width + segment_width // 2
        rgb_text = f"R:{int(color[0])}\nG:{int(color[1])}\nB:{int(color[2])}"
        draw.text((start_x-10, text_y), rgb_text, fill="black", align="center", font=font)

    # Save the final image
    # save_path = f"results/position_{BOUND_POSITION}/velocity_{BOUND_VELOCITY}/reconstructed_palette_iter_{iteration}.png"

    if method == "PSO":
        save_dir = f"results/lung4/lung_position_{BOUND_POSITION}/velocity_{BOUND_VELOCITY}"
        save_path = f"{save_dir}/lung_reconstructed_palette_iter_{iteration}.png"
    elif method == "KM":
        save_dir = "results/KM"
        save_path = f"{save_dir}/lung_reconstructed_palette_KM_max_iter_{iteration}.png"

    os.makedirs(save_dir, exist_ok=True)  # creates directories if they don’t exist
    combined_img.save(save_path, "PNG")
    print(f"Reconstructed image saved for iteration {iteration}")
    return combined_img


def PSO(n, N, max_iterations=30):
    """
    Particle Swarm Optimization for RGB palette extraction.
    n = number of palettes (particles)
    N = number of colors in each palette
    """

    # initialize velocities and positions randomly with RGB values
    velocities = np.zeros((n, N, 3))  # shape: (n_particles, n_colors_per_palette, 3)
    np.random.seed(12)  # set seed for getting the same input at each run
    palettes_xi = np.random.randint(0, 256, (n, N, 3))  # random RGB values

    # initialize best positions (local and global) to zero
    local_best = np.zeros_like(palettes_xi)  # same shape as palettes_xi but filled with zeros
    global_best = np.zeros((N, 3))  # global best initialized with zeros

    # PSO hyperparameters from lecture slides
    omega = 0.73
    alpha_1 = 1.5
    alpha_2 = 1.5

    # stopping criteria
    color_threshold = 1.0
    max_iter_stop = 0
    prev_palette = global_best.copy()  # this will save the best palette from the previous iteration

    for iteration in range(max_iterations):
        print(f"ITERATION {iteration}")

        for i in range(n):  # for each palette
            r1, r2 = np.random.rand(), np.random.rand()

            # velocity update
            velocities[i] = (
                omega * velocities[i] +
                alpha_1 * r1 * (local_best[i] - palettes_xi[i]) +
                alpha_2 * r2 * (global_best - palettes_xi[i])
            )

            # bound velocity if a bound strategy was set for it
            if BOUND_VELOCITY:
                velocities[i] = apply_bounding(velocities[i], bound_type=BOUND_VELOCITY, lower=-50, upper=50)

            # update positions
            palettes_xi[i] = palettes_xi[i] + velocities[i]

            # bound the position, position should be always bounded
            palettes_xi[i] = apply_bounding(palettes_xi[i], bound_type=BOUND_POSITION, lower=0, upper=255)

            # evaluate fitness and update local and global best
            fitness_palette_xi = evaluate_palette_fitness(palettes_xi[i])
            if fitness_palette_xi < evaluate_palette_fitness(local_best[i]):
                local_best[i] = palettes_xi[i]

            if fitness_palette_xi < evaluate_palette_fitness(global_best):
                global_best = palettes_xi[i]

            # ------------------------------------------------ check if we have to stop--------------------------------

        color_change = np.linalg.norm(global_best - prev_palette, axis=1).mean()
        prev_palette = global_best.copy()
        # print(color_change)
        if color_change < color_threshold:
            max_iter_stop+=1
            if max_iter_stop == 5:
                print(f"Stopping! Palette colors converged (Δ < {color_threshold}) color threshold")
                break
        else:
            max_iter_stop = 0

        # ------------------ visualize and save the global best every 10 iterations -----------------------------------
        if iteration % 10 == 0 or iteration == max_iterations - 1:
            cluster_recreate_image_with_palette(rgb_values_original, global_best, width, height, iteration)
        print(f"Global Best Palette (Iteration {iteration}):\n{global_best.astype(int)}\n")

    return local_best, global_best


def kmeans_quantization(N, max_iter=100):
    """
    K-means clustering to quantize an image

    n_init – Number of times the k-means algorithm is run with different centroid seeds.
    The final results is the best output of `n_init` consecutive runs in terms of inertia.
    Several runs are recommended for sparse high-dimensional problems

    max_iter – Maximum number of iterations of the k-means algorithm for a single run
    """
    kmeans = KMeans(n_clusters=N, random_state=12, n_init=10, max_iter=max_iter)
    kmeans.fit(rgb_values_original)
    palette = kmeans.cluster_centers_.astype(np.uint8)  # get palette colors

    # reconstruct image
    cluster_recreate_image_with_palette(rgb_values_original, palette, width, height, iteration=max_iter, method="KM")

    return palette


# Run PSO
local_best, global_best = PSO(n=20, N=3, max_iterations=20)
# kmeans_quantization(6, max_iter=1)
# kmeans_quantization(6, max_iter=3)
# kmeans_quantization(6, max_iter=5)
# kmeans_quantization(6, max_iter=10)
# kmeans_quantization(6, max_iter=20)
# kmeans_quantization(6, max_iter=50)
# kmeans_quantization(6, max_iter=100)
# kmeans_quantization(6, max_iter=200)






