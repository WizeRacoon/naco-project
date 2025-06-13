import os
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import manhattan_distances
from PIL import Image, ImageDraw, ImageFont
import matplotlib
import cv2

if os.environ.get('DISPLAY', '') == '':
    print('No display found. Using non-interactive Agg backend.')
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')  # or another GUI backend if needed

import matplotlib.pyplot as plt

# bounding options
BOUND_VELOCITY = "clip"  # "clip", "reflection", None
BOUND_POSITION = "clip"  # "clip", "reflection"

DISTANCE_METRIC = "L2" # Euclidean
# DISTANCE_METRIC = "L1" # Manhattan

def load_iamge(image_path):
    image_original = Image.open(image_path).convert("L")
    width, height = image_original.size
    grey_values_original = np.array(image_original.getdata())
    return grey_values_original, width, height

def evaluate_palette_fitness(palette, grey_values_original):
    if DISTANCE_METRIC == "L2":
        distances = cdist(grey_values_original[:, None], palette[:, None], metric='sqeuclidean')
    elif DISTANCE_METRIC == "L1":
        distances = manhattan_distances(grey_values_original[:, None], palette[:, None])
    else:
        raise ValueError("Unsupported distance metric. Use 'L2' for Euclidean or 'L1' for Manhattan.")
    min_distances = np.min(distances, axis=1)
    return np.sum(min_distances)

def clip_ro_rgb(values):
    return np.clip(values, 0, 255)

def apply_bounding(values, bound_type="clip", lower=0, upper=255):
    if bound_type == "clip":
        return np.clip(values, lower, upper)
    elif bound_type == "reflection":
        values = np.where(values > upper, 2 * upper - values, values)
        values = np.where(values < lower, 2 * lower - values, values)
        return np.clip(values, lower, upper)

def histogram_valley_peaks(grey_values, smoothing_kernel=5, log_transform=True, show_plot=True):
    hist, bins = np.histogram(grey_values, bins=256, range=(0, 255))
    bin_centers = (bins[:-1] + bins[1:]) / 2

    raw_hist = hist.copy()

    if log_transform:
        hist = np.log1p(hist)

    if smoothing_kernel > 1:
        kernel = np.ones(smoothing_kernel) / smoothing_kernel
        hist = np.convolve(hist, kernel, mode='same')

    valley_indices = np.where((hist[1:-1] < hist[:-2]) & (hist[1:-1] < hist[2:]))[0] + 1
    valley_intensities = bin_centers[valley_indices]

    if show_plot:
        plt.figure(figsize=(10, 4))
        plt.plot(bin_centers, raw_hist, label='Raw Histogram', color='gray', alpha=0.5)
        if log_transform:
            plt.plot(bin_centers, hist, label='Log-Smoothed Histogram', color='blue')
        else:
            plt.plot(bin_centers, hist, label='Smoothed Histogram', color='blue')
        plt.scatter(bin_centers[valley_indices], hist[valley_indices], color='red', label='Valleys')
        plt.title('Histogram with Detected Valleys')
        plt.xlabel('Gray Level')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return valley_intensities.astype(int)

def cluster_recreate_image_with_palette(grey_values_original, palette, width, height, iteration, img_name, output_directory):
    if DISTANCE_METRIC == "L2":
        distances = cdist(grey_values_original[:, None], palette[:, None], metric='sqeuclidean')
    elif DISTANCE_METRIC == "L1":
        distances = manhattan_distances(grey_values_original[:, None], palette[:, None])
    else:
        raise ValueError("Unsupported distance metric. Use 'L2' for Euclidean or 'L1' for Manhattan.")
    closest_palette_indices = np.argmin(distances, axis=1)
    clustered_image = palette[closest_palette_indices].astype(np.uint8)
    clustered_array = clustered_image.reshape((height, width))
    clustered_img = Image.fromarray(clustered_array, mode='L')

    palette_height = 50
    palette_width = width
    spacing = 0
    text_height = 40
    total_height = height + palette_height + spacing + text_height

    combined_img = Image.new("RGB", (width, total_height), (255, 255, 255))
    clustered_rgb = clustered_img.convert('RGB')
    combined_img.paste(clustered_rgb, (0, 0))

    palette_image = np.zeros((palette_height, palette_width), dtype=np.uint8)
    num_colors = len(palette)
    segment_width = palette_width // num_colors

    for i, intensity in enumerate(palette):
        start_x = i * segment_width
        end_x = (i + 1) * segment_width if i < num_colors - 1 else palette_width
        palette_image[:, start_x:end_x] = intensity

    palette_img = Image.fromarray(palette_image, mode='L').convert('RGB')
    combined_img.paste(palette_img, (0, height + spacing))

    draw = ImageDraw.Draw(combined_img)
    # font = ImageFont.truetype("arial.ttf", 8) # you can set the font type if desired (not needed)
    text_y = height + spacing + palette_height + 5

    for i, intensity in enumerate(palette):
        start_x = i * segment_width + segment_width // 2
        label = f"R:{int(intensity)}"
        draw.text((start_x-10, text_y), label, fill="black", align="center")
        # draw.text((start_x-10, text_y), label, fill="black", align="center", font=font) # setting font type if desired (not needed)

    img_name = os.path.basename(img_name)
    save_dir = f"{output_directory}/{img_name}/lung_position_{BOUND_POSITION}/velocity_{BOUND_VELOCITY}"
    save_path = f"{save_dir}/lung_reconstructed_palette_iter_{iteration}.png"

    os.makedirs(save_dir, exist_ok=True)
    combined_img.save(save_path, "PNG")
    return combined_img


from genetic_clustering import genetic_clustering

def PSO(n, N, image_path, max_iterations, output_directory, save_iteration_list, apply_ga=True):
    """
    Particle Swarm Optimization for RGB palette extraction.
    n = number of palettes (particles)
    N = number of colors in each palette
    apply_ga = whether to apply genetic clustering after PSO
    """
    grey_values_original, width, height = load_iamge(image_path)
    img_name = image_path.split(".")[0]

    velocities = np.zeros((n, N))
    np.random.seed(12)

    valley_intensities = histogram_valley_peaks(grey_values_original, smoothing_kernel=7, log_transform=True, show_plot=True)
    if len(valley_intensities) >= N:
        palette_base = np.random.choice(valley_intensities, size=(n, N), replace=True)
    else:
        print("[Warning] Not enough valley points; using random initialization instead.")
        palette_base = np.random.randint(0, 256, (n, N))

    palettes_xi = palette_base.copy()
    local_best = np.zeros_like(palettes_xi)
    global_best = np.zeros(N)

    omega = 0.73
    alpha_1 = 1.5
    alpha_2 = 1.5

    color_threshold = 1.0
    max_iter_stop = 0
    prev_palette = global_best.copy()

    all_global_bests = []  # Store global best palettes for GA

    for iteration in range(1, max_iterations + 1):
        for i in range(n):
            r1, r2 = np.random.rand(), np.random.rand()

            velocities[i] = (
                omega * velocities[i] +
                alpha_1 * r1 * (local_best[i] - palettes_xi[i]) +
                alpha_2 * r2 * (global_best - palettes_xi[i])
            )

            if BOUND_VELOCITY:
                velocities[i] = apply_bounding(velocities[i], bound_type=BOUND_VELOCITY, lower=-50, upper=50)

            palettes_xi[i] = palettes_xi[i] + velocities[i]
            palettes_xi[i] = apply_bounding(palettes_xi[i], bound_type=BOUND_POSITION, lower=0, upper=255)

            fitness_palette_xi = evaluate_palette_fitness(palettes_xi[i], grey_values_original)
            if fitness_palette_xi < evaluate_palette_fitness(local_best[i], grey_values_original):
                local_best[i] = palettes_xi[i]

            if fitness_palette_xi < evaluate_palette_fitness(global_best, grey_values_original):
                global_best = palettes_xi[i]

        color_change = np.linalg.norm(global_best - prev_palette).mean()
        prev_palette = global_best.copy()
        if color_change < color_threshold:
            max_iter_stop += 1
            if max_iter_stop == 5:
                print(f"Stopping! Palette colors converged (Δ < {color_threshold}) color threshold")
                break
        else:
            max_iter_stop = 0

        if iteration in save_iteration_list:
            cluster_recreate_image_with_palette(grey_values_original, global_best, width, height, iteration, img_name, output_directory)
            print(f"Iteration {iteration} Global Best Palette: {global_best.astype(int)} - SAVED")
        else:
            print(f"Iteration {iteration} Global Best Palette: {global_best.astype(int)}")

        all_global_bests.append(global_best)

    # Apply Genetic Clustering if enabled
    if apply_ga:
        print("Starting genetic clustering")
        unique_global_bests = np.unique(np.array(all_global_bests), axis=0)
        ga_palette = genetic_clustering(grey_values_original, initial_population=unique_global_bests, num_clusters=N)
        print(f"GA palette: {ga_palette}")
        ga_palette = np.clip(np.round(ga_palette), 0, 255).astype(np.uint8)
        print(f"Rounded GA palette: {ga_palette}")

        # Visualize and save the final result
        cluster_recreate_image_with_palette(grey_values_original, ga_palette, width, height, iteration="palette_GA", img_name=f"{img_name}_GA", output_directory=output_directory)

    return local_best, global_best


def save_overlay_pso_segments_on_xray(image_name, original_gray_img, pso_segmented_img, output_directory, iteration, min_area=100):
        """Overlays contours of PSO-segmented image onto the original grayscale X-ray."""
        h_orig, w_orig = pso_segmented_img.shape
        h_cut = h_orig - 90
        pso_segmented_img = pso_segmented_img[:h_cut, :]
        
        # Ensure both images are the same shape
        if original_gray_img.shape != pso_segmented_img.shape:
            raise ValueError("Original and PSO-segmented images must have the same shape.")
        
        # Create a color version of the grayscale image to draw in color
        overlay_img = cv2.cvtColor(original_gray_img, cv2.COLOR_GRAY2BGR)

        # Get unique intensity values (segments)
        unique_vals = np.unique(pso_segmented_img)
        
        for val in unique_vals:
            # Create binary mask
            mask = (pso_segmented_img == val).astype(np.uint8) * 255

            # Optional: clean small noise
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Filter out very small areas
            for cnt in contours:
                if cv2.contourArea(cnt) >= min_area:
                    cv2.drawContours(overlay_img, [cnt], -1, (0, 255, 0), 1)

        cv2.imwrite(f"{output_directory}/{image_name}/lung_position_clip/velocity_clip/Xray_PSO_overlay_{iteration}.png", overlay_img)
