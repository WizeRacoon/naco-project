import os
import numpy as np
from scipy.spatial.distance import cdist
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

OUTPUT_DIR_RESULTS = "data/PSO/images"
# bounding options
BOUND_VELOCITY = "clip"  # "clip", "reflection", None
BOUND_POSITION = "clip"  # "clip", "reflection"


def load_iamge(image_path):
    image_original = Image.open(image_path).convert("L")
    width, height = image_original.size
    print(f"Image size:{width}w, {height}h")
    grey_values_original = np.array(image_original.getdata())  # now this will be a 1D array
    return grey_values_original, width, height


def evaluate_palette_fitness(palette, grey_values_original):
    """Fitness score = squared Euclidean distance between pixels and palette"""
    distances = cdist(grey_values_original[:, None], palette[:, None], metric='sqeuclidean')
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


# def visualize_and_save_palette(palette, iteration):
#     """Display and save palette"""
#     fig, ax = plt.subplots(figsize=(6, 2))
#
#     for i, color in enumerate(palette):
#         rect = plt.Rectangle((i, 0), 1, 1, color=color / 255)
#         ax.add_patch(rect)
#         ax.text(i + 0.5, -0.4, f'R: {int(color[0])}', ha='center', fontsize=10)
#         ax.text(i + 0.5, -0.6, f'G: {int(color[1])}', ha='center', fontsize=10)
#         ax.text(i + 0.5, -0.8, f'B: {int(color[2])}', ha='center', fontsize=10)
#
#     ax.set_xlim(0, len(palette))
#     ax.set_ylim(0, 1)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_frame_on(False)
#
#     plt.title(f"Global Best Palette (Iteration {iteration+1})")
#     plt.savefig(f"{OUTPUT_DIR_RESULTS}/palette_iter_{iteration+1}.png", bbox_inches='tight', dpi=150)
#     plt.show()


def cluster_recreate_image_with_palette(grey_values_original, palette, width, height, iteration, img_name, output_directory):
    """
    Assigns each pixel to the closest color in the palette, reconstructs the image,
    and appends the best palette with grey labels below it before saving.
    """

    # compute Euclidean distance between each pixel and each color in the palette
    distances = cdist(grey_values_original[:, None], palette[:, None], metric='sqeuclidean')

    # for each pixel, find the index of the closest color in the palette
    closest_palette_indices = np.argmin(distances, axis=1)

    # replace each pixel with the corresponding palette color
    clustered_image = palette[closest_palette_indices].astype(np.uint8)

    clustered_array = clustered_image.reshape((height, width))
    clustered_img = Image.fromarray(clustered_array, mode='L')

    # visualisation settings
    palette_height = 50
    palette_width = width
    spacing = 0
    text_height = 40
    total_height = height + palette_height + spacing + text_height

    combined_img = Image.new("RGB", (width, total_height), (255, 255, 255))
    clustered_rgb = clustered_img.convert('RGB')
    combined_img.paste(clustered_rgb, (0, 0))

    # palette visualization
    palette_image = np.zeros((palette_height, palette_width), dtype=np.uint8)
    num_colors = len(palette)
    segment_width = palette_width // num_colors

    for i, intensity in enumerate(palette):
        start_x = i * segment_width
        end_x = (i + 1) * segment_width if i < num_colors - 1 else palette_width
        palette_image[:, start_x:end_x] = intensity

    # convert palette to PIL and put it below the reconstructed image
    palette_img = Image.fromarray(palette_image, mode='L').convert('RGB')
    combined_img.paste(palette_img, (0, height + spacing))

    # write intensity values under the palette
    draw = ImageDraw.Draw(combined_img)
    # font_size = 8
    font = ImageFont.truetype("arial.ttf", 8)

    text_y = height + spacing + palette_height + 5

    for i, intensity in enumerate(palette):
        start_x = i * segment_width + segment_width // 2
        label = f"R:{int(intensity)}"
        draw.text((start_x-10, text_y), label, fill="black", align="center", font=font)

    # Save the final image

    # img_name = img_name.split("\\")[1] # for Windows
    img_name = img_name.split("/")[3] # for Linux
    save_dir = f"{output_directory}/{img_name}/lung_position_{BOUND_POSITION}/velocity_{BOUND_VELOCITY}"
    save_path = f"{save_dir}/lung_reconstructed_palette_iter_{iteration}.png"

    os.makedirs(save_dir, exist_ok=True)  # creates directories if they don’t exist
    combined_img.save(save_path, "PNG")
    print(f"Reconstructed image saved for iteration {iteration}")
    return combined_img


def PSO(n, N, image_path, max_iterations=30, output_directory = "PSO/images/"):
    """
    Particle Swarm Optimization for RGB palette extraction.
    n = number of palettes (particles)
    N = number of colors in each palette
    """
    # load the image
    grey_values_original, width, height = load_iamge(image_path)
    img_name = image_path.split(".")[0]

    # initialize velocities and positions randomly with grey values
    velocities = np.zeros((n, N))  # shape: (n_particles, n_colors_per_palette, 3)
    np.random.seed(12)  # set seed for getting the same input at each run
    palettes_xi = np.random.randint(0, 256, (n, N))  # random grey values

    # initialize best positions (local and global) to zero
    local_best = np.zeros_like(palettes_xi)  # same shape as palettes_xi but filled with zeros
    global_best = np.zeros(N)  # global best initialized with zeros

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
            fitness_palette_xi = evaluate_palette_fitness(palettes_xi[i], grey_values_original)
            if fitness_palette_xi < evaluate_palette_fitness(local_best[i], grey_values_original):
                local_best[i] = palettes_xi[i]

            if fitness_palette_xi < evaluate_palette_fitness(global_best, grey_values_original):
                global_best = palettes_xi[i]

            # ------------------------------------------------ check if we have to stop--------------------------------

        color_change = np.linalg.norm(global_best - prev_palette).mean()
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
            cluster_recreate_image_with_palette(grey_values_original, global_best, width, height, iteration, img_name, output_directory)
        print(f"Global Best Palette (Iteration {iteration}):\n{global_best.astype(int)}\n")

    return local_best, global_best


def main():
    # Run PSO for each image in the input folder
    folder_path = 'INPUT_IMAGES'
    image_extensions = ('.png', '.jpg', '.jpeg')

    with os.scandir(folder_path) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.lower().endswith(image_extensions):
                image_path = entry.path
                print(f"Processing {entry.path}")

                local_best, global_best = PSO(n=20, N=3, image_path=entry.path, max_iterations=10)


def main_lisanne():
    # Run PSO for each image in the filtered_images folder and its subfolders
    input_folder = './input_data/filtered_images_split/'
    output_folder = './OUTPUT_IMAGES/'
    image_extensions = ('.png', '.jpg', '.jpeg')

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    print(f"Processing images in {input_folder} and saving to {output_folder}")

    for root, _, files in os.walk(input_folder):
        print(f"Processing folder: {root}")
        for file in files:
            print(f"Found file: {file}")
            if file.lower().endswith(image_extensions):
                print(f"Found image: {file}")
                # Construct the input image path
                image_path = os.path.join(root, file)
                print(f"Processing {image_path}")

                # Determine the label based on the folder structure
                if 'NORMAL' in root.upper():
                    label = 'NORMAL'
                elif 'ATELECTASIS' in root.upper():
                    label = 'ATELECTASIS'
                else:
                    print(f"Skipping {image_path} (unknown label)")
                    continue

                # Construct the output path
                relative_path = os.path.relpath(root, input_folder)
                output_subfolder = os.path.join(output_folder, relative_path)
                os.makedirs(output_subfolder, exist_ok=True)
                output_image_path = os.path.join(output_subfolder, file)

                # Run PSO and save the processed image
                try:
                    print(f"Running PSO for {image_path}...")
                    local_best, global_best = PSO(n=20, N=3, image_path=image_path, max_iterations=10)
                    print(f"PSO completed for {image_path}")

                    # Save the processed image
                    processed_image = cluster_recreate_image_with_palette(
                        grey_values_original=np.array(Image.open(image_path).convert("L").getdata()),
                        palette=global_best,
                        width=Image.open(image_path).size[0],
                        height=Image.open(image_path).size[1],
                        iteration=0,  # Example: iteration 0
                        img_name=output_image_path
                    )
                    print(f"Saved processed image to {output_image_path}")
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

#main_lisanne()