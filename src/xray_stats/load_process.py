import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import least_squares
import pywt
from skimage import restoration
from scipy.stats import skew, kurtosis
from scipy.signal import convolve2d


def get_index_number(folder_name):
    """
    Extracts the index number from a folder name following a specific naming convention.

    Parameters:
        folder_name (str): The name of the folder.

    Returns:
        int: Extracted index number from the folder name.
    """
    # Split the folder name at the "_" character
    name_parts = folder_name.split("_")
    # Get the second-to-last part of the name, which should be the index number
    index_part = name_parts[-2]
    # Remove any non-digit characters from the index part
    index_number = "".join(filter(str.isdigit, index_part))
    # Convert the index number to an integer and return it
    return int(index_number)


def get_tiff_stack(
    index, path_to_tiff_folders="/Volumes/FreeAgent GoFlex Drive/DanforthXRAYData/"
):
    """
    Get information about TIFF files in a specified directory.

    Args:
        index (int): The index of the desired tiff_stack.
        path_to_tiff_folders (str, optional): Path to the directory containing TIFF folders. Defaults to the provided path.

    Returns:
        tuple: A tuple containing the total number of images, a sorted list of TIFF files, and the path to the tiff_stack directory.
    """
    # Get a list of all folders in the path
    folders = os.listdir(path_to_tiff_folders)
    # Filter out anything that isn't a folder or '.DS_Store'
    folders_filtered = [
        folder
        for folder in folders
        if os.path.isdir(os.path.join(path_to_tiff_folders, folder))
        and folder != ".DS_Store"
    ]
    # Sort the list of folders based on the index number found in the folder name
    folders_sorted = sorted(folders_filtered, key=get_index_number)
    # Store the folders in a dictionary where the key is the index
    folders_dict = {}
    for folder in folders_sorted:
        idx = get_index_number(folder)
        folders_dict[idx] = folder

    path_to_tiff_stack = os.path.join(path_to_tiff_folders, folders_dict[index])

    # Get a list of all the TIFF files in the directory
    tiff_files = [f for f in os.listdir(path_to_tiff_stack) if f.endswith((".tif", ".tiff"))]

    # Sort the TIFF files based on the order of depth
    tiff_files_sorted = sorted(tiff_files, key=lambda f: int(os.path.splitext(f)[0][-4:]))

    # Get the total number of images to be processed
    total_images = len(tiff_files_sorted)

    return total_images, tiff_files_sorted, path_to_tiff_stack


def get_treatment(stack_index):
    """
    Get the Tillage and Fertilizer treatment for a given stack index.

    Parameters:
        stack_index (int): The index of the 3D X-ray stack for which to retrieve treatment information.

    Returns:
        str: A string containing the Tillage and Fertilizer treatments in the format "Tillage: <tillage>, Fertilizer: <fertilizer>".

    Example:
        treatment_info = get_treatment(1)
        # Output: "Tillage: NP, Fertilizer: NP"
    """
    metas = pd.read_csv("../data/meta.csv")
    return "Tillage: " + metas[metas["Sample ID"]==stack_index]["Tillage"].iloc[0] + ", Fertilizer: " + metas[metas["Sample ID"]==stack_index]["Fertilizer"].iloc[0]


def estimate_circle_params(x_positions, y_positions):
    """
    Estimates the parameters of a circle from given x and y positions.

    Parameters:
        x_positions (array-like): Array of x positions.
        y_positions (array-like): Array of y positions.

    Returns:
        tuple: A tuple containing the estimated center (xc, yc) and radius (r) of the circle.
    """

    def circle_equation(params, x, y):
        xc, yc, r = params
        return (x - xc) ** 2 + (y - yc) ** 2 - r**2

    # Set the initial guess for the circle parameters
    initial_params = [
        np.mean(x_positions),
        np.mean(y_positions),
        np.std(
            [
                np.max(x_positions) - np.min(x_positions),
                np.max(y_positions) - np.min(y_positions),
            ]
        )
        / 2,
    ]

    # Use the least squares method to fit the circle to the data
    result = least_squares(
        circle_equation, initial_params, args=(x_positions, y_positions)
    )

    # Extract the center and radius of the fitted circle
    xc, yc, r = result.x

    return xc, yc, r


def estimate_circle_from_image(path_to_tiff_stack, filename):
    """
    Estimate a circle from an image by analyzing the outer ring.

    Args:
        path_to_tiff_stack (str): The path to the directory containing the TIFF stack.
        filename (str): The filename of the TIFF image to be processed.

    Returns:
        tuple: Estimated circle center coordinates (xc, yc) and radius (r).
    """
    img = Image.open(os.path.join(path_to_tiff_stack, filename))

    # Get the height of the image
    height = img.size[1]

    # Calculate the 25th and 75th percentiles of the height
    h10 = np.percentile(np.arange(height), 10)
    h90 = np.percentile(np.arange(height), 90)

    # Initialize lists to store the coordinates of the outer ring
    outer_ring_x = []
    outer_ring_y = []

    # Generate the vertical indices
    indices = np.linspace(h10, h90, 30, dtype=int)

    # Collect exterior ring points
    for index in indices:
        # Extract the current line from the image array
        cur_line = np.array(img)[index, :]
        cur_line = np.convolve(cur_line, np.ones(10) / 10, mode="valid")

        # Find positions where adjacent differences are larger than 100
        significant_changes_indices = np.where(
            np.abs(np.diff(cur_line.astype("float"))) > 100
        )[0]
        if significant_changes_indices.size > 0:
            # Add the first and last indices to the list of coordinates that estimate the outer ring location
            outer_ring_x.append(significant_changes_indices[0])
            outer_ring_x.append(significant_changes_indices[-1])
            outer_ring_y.append(index)
            outer_ring_y.append(index)

    # Estimate the circle center and radius
    xc, yc, r = estimate_circle_params(outer_ring_x, outer_ring_y)
    return xc, yc, r


def estimate_circle_from_image_pre(img):
    """
    Estimate a circle from an image by analyzing the outer ring.

    Args:
        input_image (numpy.ndarray): The input image array.

    Returns:
        tuple: Estimated circle center coordinates (xc, yc) and radius (r).
    """
    # Get the height of the image
    height = img.shape[0]

    # Calculate the 25th and 75th percentiles of the height
    h10 = np.percentile(np.arange(height), 10)
    h90 = np.percentile(np.arange(height), 90)

    # Initialize lists to store the coordinates of the outer ring
    outer_ring_x = []
    outer_ring_y = []

    # Generate the vertical indices
    indices = np.linspace(h10, h90, 30, dtype=int)

    # Collect exterior ring points
    for index in indices:
        # Extract the current line from the image array
        cur_line = img[index, :]
        cur_line = np.convolve(cur_line, np.ones(10) / 10, mode="valid")

        # Find positions where adjacent differences are larger than 100
        significant_changes_indices = np.where(
            np.abs(np.diff(cur_line.astype("float"))) > 100
        )[0]
        if significant_changes_indices.size > 0:
            # Add the first and last indices to the list of coordinates that estimate the outer ring location
            outer_ring_x.append(significant_changes_indices[0])
            outer_ring_x.append(significant_changes_indices[-1])
            outer_ring_y.append(index)
            outer_ring_y.append(index)
    # Estimate the circle center and radius
    xc, yc, r = estimate_circle_params(outer_ring_x, outer_ring_y)
    return xc, yc, r


def masks_by_region(image, center_x, center_y, radius):
    """
    Generate masks for different regions within an image based on specified circle parameters.

    Parameters:
        image (array-like): The input image.
        center_x (float): X-coordinate of the circle's center.
        center_y (float): Y-coordinate of the circle's center.
        radius (float): Radius of the circle.

    Returns:
        tuple: A tuple containing three masks corresponding to the container, outside, and core regions of the circle.
    """
    r_core = 490
    r_plastic_inner = 615
    r_plastic_outer = 630
    r_outside = 695

    # Create a meshgrid of the x and y coordinates of the image
    y_coords, x_coords = np.mgrid[: image.shape[0], : image.shape[1]]

    # Calculate the distance of each pixel from the center of the circle
    distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
    mask_container = (distances > r_plastic_inner) & (distances < r_plastic_outer)
    mask_outside = distances > r_outside
    mask_core = distances < r_core

    return mask_container, mask_outside, mask_core


def get_masks(path_to_tiff_stack, filename):
    """
    Generate masks for different regions within an image based on circle parameters estimated from another image.

    Parameters:
        path_to_tiff_stack (str): Path to the directory containing TIFF images.
        filename (str): Name of the TIFF image file to use for estimating circle parameters.

    Returns:
        tuple: A tuple containing three masks corresponding to the container, outside, and core regions of the circle.
    """
    # Estimate circle parameters
    xc, yc, r = estimate_circle_from_image(path_to_tiff_stack, filename)

    # Load the image using PIL
    img = Image.open(os.path.join(path_to_tiff_stack, filename))
    img_array = np.array(img)

    # Generate masks based on circle parameters
    mask_container, mask_outside, mask_core = masks_by_region(img_array, xc, yc, r)
    return mask_container, mask_outside, mask_core


def get_masks_pre(img):
    """
    Generate masks for different regions within an image based on circle parameters estimated from the image.

    Parameters:
        img (numpy.ndarray): Input image as a NumPy array.

    Returns:
        tuple: A tuple containing three masks corresponding to the container, outside, and core regions of the circle.
    """
    # Estimate circle parameters
    xc, yc, r = estimate_circle_from_image_pre(img)
    # Generate masks based on circle parameters
    mask_container, mask_outside, mask_core = masks_by_region(img, xc, yc, r)
    return mask_container, mask_outside, mask_core


def pixels_by_region(img, xc, yc, r):
    """
    Extract pixels from different regions within an image based on circle parameters.

    Parameters:
        img (numpy.ndarray): Input image as a NumPy array.
        xc (float): X-coordinate of the circle's center.
        yc (float): Y-coordinate of the circle's center.
        r (float): Radius of the circle.

    Returns:
        tuple: A tuple containing pixel arrays for the container, outside, and core regions of the circle.
    """
    # Generate masks based on circle parameters
    mask_container, mask_outside, mask_core = masks_by_region(img, xc, yc, r)

    # Extract pixels based on masks
    img_container = img[mask_container]
    img_outside = img[mask_outside]
    img_core = img[mask_core]

    return img_container, img_outside, img_core


def denoise_wavelet(img):
    """
    Apply wavelet-based denoising to an input image.

    Parameters:
        img (numpy.ndarray): Input image as a NumPy array.

    Returns:
        numpy.ndarray: Denoised image after wavelet denoising.
    """
    # Standard deviation from plastic ring and outer portions (places that should have uniform density)
    noise_std = 2500
    # Set the wavelet denoising threshold
    threshold = noise_std * 2

    # Apply wavelet denoising
    coeffs = pywt.dwt2(img, "db2")
    coeffs_thresh = [pywt.threshold(c, threshold, "soft") for c in coeffs]
    img_denoised = pywt.idwt2(coeffs_thresh, "db2")
    # Crop the denoised image to the same size as the original
    img_denoised = np.array(img_denoised[: img.shape[0], : img.shape[1]])
    return img_denoised


def denoise_tv(img):
    """
    Apply total variation denoising to an input image.

    Parameters:
        img (numpy.ndarray): Input image as a NumPy array.

    Returns:
        numpy.ndarray: Denoised image after total variation denoising.
    """
    img_denoised = (
        restoration.denoise_tv_chambolle(img.astype("<u2"), weight=0.0075) * 65535
    )
    return img_denoised


def sliding_stats(image, window_size, show_progress, skip=1, mask=None):
    """
    Calculates the skewness, kurtosis and variance of a sliding window over an image, taking into account a mask that sets the value to NaN for any window containing masked pixels.

    Parameters:
        image (numpy.ndarray): The image data as a 2D numpy array.
        window_size (int): The size of the sliding window.
        skip (int, optional): The number of rows and columns to skip between windows. Defaults to 1.
        mask (numpy.ndarray, optional): A boolean mask the same shape as the image that specifies which pixels should be masked. Defaults to None.

    Returns:
        numpy.ndarray: A 2D numpy array of the same shape as the input image containing the skewness values for each sliding window.
    """
    # Pad the image with NaN values to handle edges
    image_padded = np.pad(
        image, window_size // 2, mode="constant", constant_values=np.nan
    )

    # Calculate the dimensions of the resulting skewness matrix
    num_rows = (image.shape[0] + skip - 1) // skip
    num_cols = (image.shape[1] + skip - 1) // skip

    # Initialize an empty array to hold the skewness values
    skewness_matrix = np.zeros((num_rows, num_cols))
    kurtosis_matrix = np.zeros((num_rows, num_cols))
    variance_matrix = np.zeros((num_rows, num_cols))

    # Iterate over the image using a sliding window
    for i in range(0, image.shape[0], skip):
        for j in range(0, image.shape[1], skip):
            # Extract the window from the padded image
            window = image_padded[i : i + window_size, j : j + window_size]

            # Check if the window contains any masked pixels
            if mask is not None and np.any(
                ~mask[i : i + window_size, j : j + window_size]
            ):
                # Set the skewness value to NaN if the window contains masked pixels
                skewness_matrix[i // skip, j // skip] = np.nan
                kurtosis_matrix[i // skip, j // skip] = np.nan
                variance_matrix[i // skip, j // skip] = np.nan
            else:
                # Calculate the skewness of the window and store it in the skewness matrix
                skewness_matrix[i // skip, j // skip] = skew(window.flatten())
                kurtosis_matrix[i // skip, j // skip] = kurtosis(window.flatten())
                variance_matrix[i // skip, j // skip] = np.var(window.flatten())
        if show_progress:
            print_progress(i, image.shape[0])
    return skewness_matrix, kurtosis_matrix, variance_matrix


def sobel_edge_detection(image):
    """
    Apply Sobel edge detection to an input image.

    Parameters:
        image (numpy.ndarray): Input image as a NumPy array.

    Returns:
        numpy.ndarray: Image showing edges detected by the Sobel operator.
    """
    # Define the Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Convert the image to grayscale if it's in color
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)

    # Convolve the image with the Sobel kernels to get the horizontal and vertical gradients
    grad_x = convolve2d(image, sobel_x, mode="same")
    grad_y = convolve2d(image, sobel_y, mode="same")

    # Compute the magnitude of the gradient at each pixel
    mag = np.sqrt(grad_x**2 + grad_y**2)

    # Normalize the magnitude to 0-255 and convert it to an 8-bit integer
    mag = (mag / np.max(mag)) * 255
    mag = mag.astype(np.uint8)

    return mag


def print_and_flush(text):
    """
    Print text to the console and flush the output, keeping the cursor at the same line.

    Parameters:
        text (str): The text to be printed.

    Returns:
        None
    """
    print(text, end="\r", flush=True)


def print_progress(current, total):
    """
    Print the progress of a task as a percentage completion.

    Parameters:
        current (int): The current progress value.
        total (int): The total value indicating completion.

    Returns:
        None
    """
    percentage_complete = (current / total) * 100
    print_and_flush(
        f"Processing image {current}/{total} ({percentage_complete:.2f}% complete)"
    )


def get_stats(
    tiff_index,
    stack_index,
    window_size,
    skip,
    denoise,
    display_progress,
    tiff_files_sorted,
    path_to_tiff_stack,
):
    """
    Calculate various statistics from an image.

    Args:
        tiff_index (int): Index of the TIFF image.
        stack_index (int): Index of the stack.
        window_size (int): Window size for sliding statistics.
        skip (int): Number of pixels to skip during sliding statistics.
        denoise (bool): Flag indicating whether to apply denoising.
        display_progress (bool): Flag indicating whether to display progress.
        tiff_files_sorted (list): Sorted list of TIFF file names.
        path_to_tiff_stack (str): Path to the TIFF stack directory.

    Returns:
        pd.DataFrame: DataFrame containing calculated statistics.
    """
    metas = pd.read_csv("../data/meta.csv")
    meta_index = metas[metas["Sample ID"] == stack_index].index[0]

    # Load the image using PIL
    img = Image.open(os.path.join(path_to_tiff_stack, tiff_files_sorted[tiff_index]))
    mask_container, mask_outside, mask_core = get_masks(
        path_to_tiff_stack, tiff_files_sorted[tiff_index]
    )
    # Convert the image to a numpy array
    img_array = np.array(img)

    if denoise:
        # If denoise flag is enabled, apply Total Variation (TV) denoising to the image
        img_array = denoise_tv(img_array)

    # Extract image data based on different masks
    img_core = img_array[mask_core]  # Extract data from the core region
    img_outer_mean = np.nanmean(
        img_array[mask_outside]
    )  # Calculate mean of the outer region
    img_container_mean = np.nanmean(
        img_array[mask_container]
    )  # Calculate mean of the container region

    # Calculate sliding window statistics for the image using given parameters
    skews, kurts, varis = sliding_stats(
        img_array, window_size, display_progress, skip, mask_core
    )

    # Detect edges using the Sobel edge detection algorithm and filter using core mask
    edges = sobel_edge_detection(img_array)
    edges = edges[mask_core]

    data_dict = {
        "stack_index": stack_index,
        "tiff_index": tiff_index,
        "file_name": tiff_files_sorted[tiff_index],
        "tillage": metas.loc[meta_index, "Tillage"],
        "fertilizer": metas.loc[meta_index, "Fertilizer"],
        "tillage-fertilizer": metas.loc[meta_index, "Tillage"]
        + "-"
        + metas.loc[meta_index, "Fertilizer"],
        "block": metas.loc[meta_index, "Block"],
        "sub-rep": metas.loc[meta_index, "Sub-rep"],
        "window_size": window_size,
        "skip": skip,
        "denoise": denoise,
        "skew_mean": np.nanmean(skews),
        "skew_median": np.nanmedian(skews),
        "skew_std": np.nanstd(skews),
        "skew_p5": np.percentile(np.array(skews[~np.isnan(skews)]), 5),
        "skew_p95": np.percentile(np.array(skews[~np.isnan(skews)]), 95),
        "kurt_mean": np.nanmean(kurts),
        "kurt_median": np.nanmedian(kurts),
        "kurt_std": np.nanstd(kurts),
        "kurt_p5": np.percentile(np.array(kurts[~np.isnan(kurts)]), 5),
        "kurt_p95": np.percentile(np.array(kurts[~np.isnan(kurts)]), 95),
        "vari_mean": np.nanmean(varis),
        "vari_median": np.nanmedian(varis),
        "vari_std": np.nanstd(varis),
        "vari_p5": np.percentile(np.array(varis[~np.isnan(varis)]), 5),
        "vari_p95": np.percentile(np.array(varis[~np.isnan(varis)]), 95),
        "edge_mean": np.nanmean(edges),
        "edge_median": np.nanmedian(edges),
        "edge_std": np.nanstd(edges),
        "edge_p5": np.percentile(np.array(edges), 5),
        "edge_p95": np.percentile(np.array(edges), 95),
        "img_mean": np.nanmean(img_core),
        "img_median": np.nanmedian(img_core),
        "img_std": np.nanstd(img_core),
        "img_p5": np.percentile(np.array(img_core), 5),
        "img_p95": np.percentile(np.array(img_core), 95),
        "img_mean_norm (g/cm3)": (np.nanmean(img_core) - img_outer_mean)
        / (img_container_mean - img_outer_mean)
        * 1.022,
        "img_median_norm (g/cm3)": (np.nanmedian(img_core) - img_outer_mean)
        / (img_container_mean - img_outer_mean)
        * 1.022,
        "img_std_norm (g/cm3)": np.nanstd(img_core)
        / (img_container_mean - img_outer_mean)
        * 1.022,
        "img_p5_norm (g/cm3)": (np.percentile(np.array(img_core), 5) - img_outer_mean)
        / (img_container_mean - img_outer_mean)
        * 1.022,
        "img_p95_norm (g/cm3)": (np.percentile(np.array(img_core), 95) - img_outer_mean)
        / (img_container_mean - img_outer_mean)
        * 1.022,
        "depth": tiff_index * 0.0039,
    }

    # create a DataFrame from the dictionary
    df_row = pd.DataFrame([data_dict])
    return df_row


def visualize_sliding_window_statistics(
    tiff_index, stack_index, window_size, skip, denoise, show_progress
):
    """
    Visualize various image statistics for a specific TIFF image in a 3D X-ray stack.

    This function processes a specific TIFF image from a 3D X-ray stack, calculates
    statistics such as skewness, kurtosis, variances, and edge detection, and displays
    them using subplots in a single figure.

    Args:
        tiff_index (int): Index of the TIFF image to process.
        stack_index (int): Index of the 3D X-ray stack.
        window_size (int): Size of the sliding window for statistics calculation.
        skip (int): Number of pixels to skip in sliding window.
        denoise (bool): Flag indicating whether to apply denoising to the image.
        show_progress (bool): Flag indicating whether to display progress indicators.

    Returns:
        numpy.ndarray, numpy.ndarray, numpy.ndarray: Arrays containing skewness, kurtosis, and variances.
    """
    # Obtain image stack information
    total_images, tiff_files_sorted, path_to_tiff_stack = get_tiff_stack(stack_index)

    # Create the figure and axis objects for the plot
    fig, ax = plt.subplots(3, 2, figsize=(10, 15))

    # Load the image using PIL
    img = Image.open(os.path.join(path_to_tiff_stack, tiff_files_sorted[tiff_index]))

    # Obtain masks for different regions
    mask_container, mask_outside, mask_core = get_masks(
        path_to_tiff_stack, tiff_files_sorted[tiff_index]
    )

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Apply denoising if specified
    if denoise:
        img_array = denoise_tv(img_array)

    # Calculate statistics
    skews, kurts, varis = sliding_stats(img_array, window_size, show_progress, skip)
    edges = sobel_edge_detection(img_array)

    # Update the plot with the new images and titles
    ax[0, 0].imshow(skews, cmap="gray", vmin=-2, vmax=5)
    ax[0, 0].set_title("Skewness: " + tiff_files_sorted[tiff_index])
    ax[1, 0].imshow(img_array, cmap="gray", vmin=0, vmax=17000)
    ax[1, 0].set_title("Image: " + tiff_files_sorted[tiff_index])
    ax[2, 0].imshow(kurts, cmap="gray", vmin=-2, vmax=10)
    ax[2, 0].set_title("Kurtosis: " + tiff_files_sorted[tiff_index])
    ax[0, 1].imshow(varis, cmap="gray", vmin=0, vmax=2000000)
    ax[0, 1].set_title("Variances: " + tiff_files_sorted[tiff_index])
    ax[1, 1].imshow(edges, cmap="gray", vmin=0, vmax=100)
    ax[1, 1].set_title("Sobel Edge: " + tiff_files_sorted[tiff_index])
    plt.suptitle("Sample of " + get_treatment(stack_index))
    fig.tight_layout()
    return skews, kurts, varis


def display_window_with_histogram(
    tiff_index, stack_index, window_size, denoise, px, py
):
    """
    Display a windowed region of a TIFF image along with a histogram.

    This function loads a specific TIFF image from a 3D X-ray stack, extracts a windowed region,
    and displays the window along with its histogram. Optionally, denoising can be applied.

    Args:
        tiff_index (int): Index of the TIFF image within the stack.
        stack_index (int): Index of the 3D X-ray stack.
        window_size (int): Size of the window.
        denoise (bool): Flag indicating whether to apply denoising to the image.
        px (int): X-coordinate of the center of the window.
        py (int): Y-coordinate of the center of the window.
        skews (numpy.ndarray): Array containing skewness values for the image.
        kurts (numpy.ndarray): Array containing kurtosis values for the image.
    """
    total_images, tiff_files_sorted, path_to_tiff_stack = get_tiff_stack(stack_index)

    # Create the figure and axis objects for the plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Load the image using PIL
    img = Image.open(os.path.join(path_to_tiff_stack, tiff_files_sorted[tiff_index]))

    # Convert the image to a numpy array
    img_array = np.array(img)

    if denoise:
        img_array = denoise_tv(img_array)

    window = img_array[
        (py - window_size // 2) : (py + window_size // 2),
        (px - window_size // 2) : (px + window_size // 2),
    ]

    # Update the plot with the new image and histogram
    ax[0].imshow(window, cmap="gray", vmin=0, vmax=10000)
    ax[1].hist(window.ravel(), bins=100)

    # Calculate kurtosis and skew for window
    skewz = skew(window.flatten())
    kurtz = kurtosis(window.flatten())
    # Set the title for the entire figure
    fig.suptitle(f"Kurtosis: {kurtz}, Skew: {skewz}", fontsize=20)

    # Adjust the space between the subplots and the main title
    fig.subplots_adjust(top=0.85)


def display_horizontal_xray_slice(tiff_index, stack_index, max_intensity=10000, figure_size = 7):
    """
    Display a single TIFF image from a 3D X-ray stack.

    Parameters:
        tiff_index (int): Index of the TIFF image within the stack.
        stack_index (int): Index of the 3D X-ray stack.
        max_intensity (int, optional): Maximum pixel intensity for histogram (default is 10000).
        figure_size (int, optional): Size of the displayed image (default is 7).

    Returns:
        np.ndarray: Numpy array representation of the displayed image.

    Displays a specific TIFF image from a 3D X-ray stack using matplotlib. The function loads the specified
    TIFF image, converts it to a numpy array, and displays it with a title indicating the stack and TIFF index.

    Example:
        img = display_horizontal_xray_slice(5, 1, max_intensity=12000, figure_size=8)
    """
    total_images, tiff_files_sorted, path_to_tiff_stack = get_tiff_stack(stack_index)

    # Create the figure and axis objects for the plot
    fig, ax = plt.subplots(figsize=(figure_size, figure_size))

    # Load the image using PIL
    img = Image.open(os.path.join(path_to_tiff_stack, tiff_files_sorted[tiff_index]))

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Update the plot with the new image and title
    ax.imshow(img_array, cmap="gray", vmin=0, vmax=max_intensity)

    # Set the title for the entire figure
    ax.set_title(f"Stack Index: {stack_index}, Tiff Index: {tiff_index} ("+get_treatment(stack_index)+ ")", fontsize=20)

    return img_array


def display_vertical_xray_slice(stack_index, target_row, figure_size = 7):
    """
    Extract a vertical image slice from a 3D X-ray stack for a specified row.

    Parameters:
        stack_index (int): Index of the 3D X-ray stack.
        target_row (int): The row index for which the vertical slice is extracted.
        figure_size (int, optional): Size of the displayed image (default is 7).

    Returns:
        numpy.ndarray: Vertical image slice for the specified row.

    This function processes a 3D X-ray image stack and extracts a vertical image slice
    from the stack at the given row index. It performs vertical scaling based on background noise
    and container density to enhance the visualization of the vertical slice.

    Example:
        vertical_slice = display_vertical_xray_slice(1, 100, figure_size=8)
    """

    # Obtain X-ray stack information
    total_images, tiff_files_sorted, path_to_tiff_stack = get_tiff_stack(stack_index)

    # Initialize lists to store background noise and container density values
    outer_noise_attenuation = []
    container_attenuation = []
    i = 0

    # Subsample the X-ray slices to calculate average background noise and container density
    for tiff_index in np.linspace(0, total_images - 1, num=50, dtype=int):
        i += 1
        image = Image.open(
            os.path.join(path_to_tiff_stack, tiff_files_sorted[tiff_index])
        )
        image_array = np.array(image)
        mask_container, mask_outside, _ = get_masks_pre(image_array)

        outer_noise_attenuation.append(np.nanmean(image_array[mask_outside]))
        container_attenuation.append(np.nanmean(image_array[mask_container]))
        print_progress(i, 50)

    # Calculate medians for background noise and container density
    outer_median = np.median(outer_noise_attenuation)
    container_median = np.median(container_attenuation)

    # Extract specified row from each X-ray slice and perform vertical scaling
    row_slice = []
    for tiff_index in range(total_images):
        image = Image.open(
            os.path.join(path_to_tiff_stack, tiff_files_sorted[tiff_index])
        )
        image_array = np.array(image)
        row_slice.append(image_array[target_row])
        print_progress(tiff_index, total_images)

    # Perform vertical scaling using the calculated medians
    row_slice = (row_slice - outer_median) / (container_median - outer_median) * 1.022

    # Create the figure and axis objects for the plot
    fig, ax = plt.subplots(figsize=(figure_size, figure_size))

    # Update the plot with the new image and title
    ax.imshow(row_slice, cmap="gray", vmin=0, vmax=np.percentile(np.array(row_slice), 95))

    # Set the title for the entire figure
    ax.set_title(f"Stack Index: {stack_index} ("+get_treatment(stack_index)+ ")", fontsize=20)
    return row_slice


def intensity_histogram_by_depth(stack_index, tiff_skips, slice_thickness_cm=0.0039, max_intensity=17000):
    """
    Generate and display heatmaps of logarithmic histogram counts of x-ray intensities for different depths.

    Parameters:
        stack_index (int): The index of the stack.
        tiff_skips (int): The number of slices to skip between histogram calculations.
        slice_thickness_cm (float, optional): Thickness of each slice in cm (default is 39 microns).
        max_intensity (int, optional): Maximum pixel intensity for histogram (default is 17000).

    Returns:
        None (displays the heatmaps)
    """
    total_images, tiff_files_sorted, path_to_tiff_stack = get_tiff_stack(stack_index)
    xc, yc, r = estimate_circle_from_image(path_to_tiff_stack, tiff_files_sorted[0])

    # Initialize lists to store histograms for different regions
    hists_core = []
    hists_container = []
    hists_outside = []

    # Calculate y-axis tick positions based on slice thickness
    y_depth_positions = np.arange(
        0, total_images * slice_thickness_cm, tiff_skips * slice_thickness_cm
    )

    tiffs = np.arange(0, total_images, tiff_skips)
    for tiff in tiffs:
        # Get histograms for all three regions
        (
            counts_core,
            counts_container,
            counts_outside,
            bin_edges,
            _,
            _,
        ) = get_image_histogram(tiff, tiff_files_sorted, path_to_tiff_stack, xc, yc, r, max_intensity)
        hists_core.append(counts_core)
        hists_container.append(counts_container)
        hists_outside.append(counts_outside)
        print_progress(tiff, max(tiffs))
    # Add a small constant value (e.g., 1e-1, which is less than one - the lowest non-zero count)
    # to the histogram counts to avoid division by zero
    epsilon = 1e-1

    # Create subplots for the heatmaps of different regions
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Create heatmaps for core, container, and outside regions
    im_core = axs[0].imshow(
        np.log2(np.array(hists_core) + epsilon),
        cmap="viridis",
        interpolation="nearest",
        aspect="auto",
    )
    axs[0].set_title("Core Region Histogram Counts")

    im_container = axs[1].imshow(
        np.log2(np.array(hists_container) + epsilon),
        cmap="viridis",
        interpolation="nearest",
        aspect="auto",
    )
    axs[1].set_title("Container Region Histogram Counts")

    im_outside = axs[2].imshow(
        np.log2(np.array(hists_outside) + epsilon),
        cmap="viridis",
        interpolation="nearest",
        aspect="auto",
    )
    axs[2].set_title("Outside Region Histogram Counts")

    # Set common colorbar for all subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im_outside, cax=cbar_ax)
    cbar.ax.set_ylabel("Log_2 Counts", rotation=-90, va="bottom")

    # Set common labels for all subplots
    for ax in axs:
        ax.set_xlabel("X-ray Intensity")
        ax.set_ylabel("Depth (cm)")

    # Set x-axis tick positions and labels using bin edges
    x_bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    x_tick_positions = np.arange(0, len(x_bin_centers), len(x_bin_centers) // 6)
    x_tick_labels = [
        f"{int(bin_center)}" for bin_center in x_bin_centers[x_tick_positions]
    ]
    for ax in axs:
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(x_tick_labels)

    # Set y-axis tick positions and labels for depth in cm
    y_tick_positions = np.arange(0, len(y_depth_positions), len(y_depth_positions) // 6)
    y_tick_labels = [f"{depth:.2f} cm" for depth in y_depth_positions[y_tick_positions]]
    for ax in axs:
        ax.set_yticks(y_tick_positions)
        ax.set_yticklabels(y_tick_labels)
    plt.suptitle("Sample of " + get_treatment(stack_index))
    # Show the subplots
    plt.show()


def get_image_histogram(tiff_index, tiff_files_sorted, path_to_tiff_stack, xc, yc, r, max_intensity=17000):
    """
    Calculate histogram counts and bin edges for specific regions of an X-ray slice.

    This function computes histograms for the core, container, and outside regions of an X-ray image slice.
    It also returns the corresponding bin edges for each region.

    Parameters:
        tiff_index (int): The index of the TIFF image.
        tiff_files_sorted (list): Sorted list of TIFF filenames.
        path_to_tiff_stack (str): Path to the TIFF stack.
        xc (float): X-coordinate of the estimated circle center.
        yc (float): Y-coordinate of the estimated circle center.
        r (float): Estimated radius of the circle.
        max_intensity (int, optional): Maximum pixel intensity for histogram (default is 17000).

    Returns:
        counts_core (np.ndarray): Histogram counts for the core region.
        counts_container (np.ndarray): Histogram counts for the container region.
        counts_outside (np.ndarray): Histogram counts for the outside region.
        bin_edges_core (np.ndarray): Bin edges for the core region histogram.
        bin_edges_container (np.ndarray): Bin edges for the container region histogram.
        bin_edges_outside (np.ndarray): Bin edges for the outside region histogram.
    """
    # Load the image using PIL
    img = Image.open(os.path.join(path_to_tiff_stack, tiff_files_sorted[tiff_index]))
    mask_container, mask_outside, mask_core = get_masks(
        path_to_tiff_stack, tiff_files_sorted[tiff_index]
    )
    # Convert the image to a numpy array
    img_array = np.array(img)

    img_container, img_outside, img_core = pixels_by_region(img_array, xc, yc, r)
    counts_core, bin_edges_core = np.histogram(
        img_core.ravel(), bins=256, range=(0, max_intensity)
    )
    counts_container, bin_edges_container = np.histogram(
        img_container.ravel(), bins=256, range=(0, max_intensity)
    )
    counts_outside, bin_edges_outside = np.histogram(
        img_outside.ravel(), bins=256, range=(0, max_intensity)
    )

    return (
        counts_core,
        counts_container,
        counts_outside,
        bin_edges_core,
        bin_edges_container,
        bin_edges_outside,
    )


def plot_intensity_stats_by_depth(stack_index, skip):
    """
    Plot intensity statistics for different regions by depth in a 3D X-ray stack.

    Parameters:
        stack_index (int): Index of the 3D X-ray stack.
        skip (int): Number of slices to skip between calculations.

    Returns:
        (list, list, list, list, list, list, list, list, list, list, list, list, list, list, list): 
        Lists of intensity statistics for the soil core, plastic container, and outside of the container,
        respectively. Each list contains percentiles (1st, 50th, 99th), average, and standard deviation values.

    This function calculates intensity statistics (percentiles, mean, and standard deviation)
    for different regions (soil core, plastic container, and outside of the container)
    by depth in a 3D X-ray stack and plots the results.

    Example:
        basic_stats_by_depth = plot_intensity_stats_by_depth(1, 5)
    """

    total_images, tiff_files_sorted, path_to_tiff_stack = get_tiff_stack(stack_index)

    # Create a list to store the percentile values
    percentiles1_core = []
    percentiles50_core = []
    percentiles99_core = []
    means_core = []
    stds_core = []

    percentiles1_container = []
    percentiles50_container = []
    percentiles99_container = []
    means_container = []
    stds_container = []

    percentiles1_outside = []
    percentiles50_outside = []
    percentiles99_outside = []
    means_outside = []
    stds_outside = []
    inds = []

    # Loop through the TIFF files and calculate the percentiles for each image
    for i, filename in enumerate(tiff_files_sorted):
        if i % skip != 0:
            continue  # Skip every skip-th file
        inds.append(i)
        basic_stats = get_intensity_stats(path_to_tiff_stack, filename)

        percentiles1_core.append(basic_stats[0])
        percentiles50_core.append(basic_stats[1])
        percentiles99_core.append(basic_stats[2])
        means_core.append(basic_stats[3])
        stds_core.append(basic_stats[4])

        percentiles1_container.append(basic_stats[5])
        percentiles50_container.append(basic_stats[6])
        percentiles99_container.append(basic_stats[7])
        means_container.append(basic_stats[8])
        stds_container.append(basic_stats[9])

        percentiles1_outside.append(basic_stats[10])
        percentiles50_outside.append(basic_stats[11])
        percentiles99_outside.append(basic_stats[12])
        means_outside.append(basic_stats[13])
        stds_outside.append(basic_stats[14])

        # Print the progress on the same line and update it
        print_progress(i + 1, total_images)

    inds = np.array(inds) * 0.0035  # Convert to depth in cm
    # Create a plot for each section of its stats as a function of their tiff index
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(inds, percentiles1_core, color="red", label="1st Percentile")
    ax[0].plot(inds, percentiles50_core, color="green", label="50th Percentile")
    ax[0].plot(inds, percentiles99_core, color="blue", label="99th Percentile")
    ax[0].plot(inds, means_core, color="black", label="Average")
    ax[0].plot(inds, stds_core, color="purple", label="Standard Deviation")
    ax[0].set_xlabel("Depth (cm)")
    ax[0].set_ylabel("Intensity")
    ax[0].set_title("Soil Core")
    ax[0].set_ylim([0, 13000])

    ax[1].plot(inds, percentiles1_container, color="red", label="1st Percentile")
    ax[1].plot(inds, percentiles50_container, color="green", label="50th Percentile")
    ax[1].plot(inds, percentiles99_container, color="blue", label="99th Percentile")
    ax[1].plot(inds, means_container, color="black", label="Average")
    ax[1].plot(inds, stds_container, color="purple", label="Standard Deviation")
    ax[1].set_xlabel("Depth (cm)")
    ax[1].set_title("Plastic Container")
    ax[1].legend()
    ax[1].set_ylim([0, 13000])

    ax[2].plot(inds, percentiles1_outside, color="red", label="1st Percentile")
    ax[2].plot(inds, percentiles50_outside, color="green", label="50th Percentile")
    ax[2].plot(inds, percentiles99_outside, color="blue", label="99th Percentile")
    ax[2].plot(inds, means_outside, color="black", label="Average")
    ax[2].plot(inds, stds_outside, color="purple", label="Standard Deviation")
    ax[2].set_xlabel("Depth (cm)")
    ax[2].set_title("Outside of Container")
    ax[2].set_ylim([0, 13000])

    plt.suptitle("Sample of " + get_treatment(stack_index))
    plt.show()

    return (
        percentiles1_core,
        percentiles50_core,
        percentiles99_core,
        means_core,
        stds_core,
        percentiles1_container,
        percentiles50_container,
        percentiles99_container,
        means_container,
        stds_container,
        percentiles1_outside,
        percentiles50_outside,
        percentiles99_outside,
        means_outside,
        stds_outside,
    )


def plot_soil_density_stats_by_depth(stack_index, skip):
    """
    Plot soil density statistics by depth for a given X-ray stack.

    Parameters:
        stack_index (int): Index of the 3D X-ray stack.
        skip (int): Number of slices to skip between statistics calculations.

    Returns:
        Tuple: A tuple containing arrays of normalized density statistics (1st Percentile, 50th Percentile,
        99th Percentile, Average, Standard Deviation) for the soil core.

    This function calculates and plots soil density statistics by depth for a specific 3D X-ray stack.
    It extracts soil density statistics for the soil core region, normalizes them, and plots the results
    as a function of depth in centimeters.

    Args:
        stack_index (int): Index of the 3D X-ray stack.
        skip (int): Number of slices to skip between statistics calculations.

    Returns:
        Tuple: A tuple containing arrays of normalized density statistics (1st Percentile, 50th Percentile,
        99th Percentile, Average, Standard Deviation) for the soil core.

    Example:
        stats = plot_soil_density_stats_by_depth(1, 5)
    """
    total_images, tiff_files_sorted, path_to_tiff_stack = get_tiff_stack(stack_index)

    # Create a list to store the percentile values
    percentiles1_core = []
    percentiles50_core = []
    percentiles99_core = []
    means_core = []
    stds_core = []

    percentiles1_container = []
    percentiles50_container = []
    percentiles99_container = []
    means_container = []
    stds_container = []

    percentiles1_outside = []
    percentiles50_outside = []
    percentiles99_outside = []
    means_outside = []
    stds_outside = []
    inds = []

    # Loop through the TIFF files and calculate the percentiles for each image
    for i, filename in enumerate(tiff_files_sorted):
        if i % skip != 0:
            continue  # Skip every skip-th file
        inds.append(i)
        basic_stats = get_intensity_stats(path_to_tiff_stack, filename)

        percentiles1_core.append(basic_stats[0])
        percentiles50_core.append(basic_stats[1])
        percentiles99_core.append(basic_stats[2])
        means_core.append(basic_stats[3])
        stds_core.append(basic_stats[4])

        percentiles1_container.append(basic_stats[5])
        percentiles50_container.append(basic_stats[6])
        percentiles99_container.append(basic_stats[7])
        means_container.append(basic_stats[8])
        stds_container.append(basic_stats[9])

        percentiles1_outside.append(basic_stats[10])
        percentiles50_outside.append(basic_stats[11])
        percentiles99_outside.append(basic_stats[12])
        means_outside.append(basic_stats[13])
        stds_outside.append(basic_stats[14])

        # Print the progress on the same line and update it
        print_progress(i + 1, total_images)

    inds = np.array(inds) * 0.0035  # Convert to depth in cm

    # Convert soil core intensities into densities.
    percentiles1_core = np.array(percentiles1_core)
    percentiles50_core = np.array(percentiles50_core)
    percentiles99_core = np.array(percentiles99_core)
    means_core = np.array(means_core)
    stds_core = np.array(stds_core)

    means_outside = np.array(means_outside)
    means_container = np.array(means_container)

    percentiles1_core_norm = (percentiles1_core - means_outside)/(means_container - means_outside)*1.022
    percentiles50_core_norm = (percentiles50_core - means_outside)/(means_container - means_outside)*1.022
    percentiles99_core_norm = (percentiles99_core - means_outside)/(means_container - means_outside)*1.022
    means_core_norm = (means_core - means_outside)/(means_container - means_outside)*1.022
    stds_core_norm = (stds_core - means_outside)/(means_container - means_outside)*1.022

    # Create a plot for each section of its stats as a function of their tiff index
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(inds, percentiles1_core_norm, color="red", label="1st Percentile")
    ax.plot(inds, percentiles50_core_norm, color="green", label="50th Percentile")
    ax.plot(inds, percentiles99_core_norm, color="blue", label="99th Percentile")
    ax.plot(inds, means_core_norm, color="black", label="Average")
    ax.plot(inds, stds_core_norm, color="purple", label="Standard Deviation")
    ax.set_xlabel("Depth (cm)")
    ax.set_ylabel("Density (g/ml)")
    ax.set_title("Soil Core Sample of " + get_treatment(stack_index))
    ax.legend()
    ax.set_ylim([0, 3])

    plt.show()

    return (
        percentiles1_core_norm,
        percentiles50_core_norm,
        percentiles99_core_norm,
        means_core_norm,
        stds_core_norm,
    )


def get_intensity_stats(path_to_tiff_stack, filename):
    """
    Calculate basic statistics of pixel intensities for different regions in an X-ray image.

    Parameters:
        path_to_tiff_stack (str): Path to the directory containing TIFF images.
        filename (str): Name of the TIFF image file.

    Returns:
        Tuple[float]: A tuple containing various intensity statistics for different regions:
            - p1_core (float): 1st percentile intensity in the core region.
            - p50_core (float): 50th percentile (median) intensity in the core region.
            - p99_core (float): 99th percentile intensity in the core region.
            - m_core (float): Mean intensity in the core region.
            - std_core (float): Standard deviation of intensity in the core region.
            - p1_container (float): 1st percentile intensity in the plastic container region.
            - p50_container (float): 50th percentile (median) intensity in the plastic container region.
            - p99_container (float): 99th percentile intensity in the plastic container region.
            - m_container (float): Mean intensity in the plastic container region.
            - std_container (float): Standard deviation of intensity in the plastic container region.
            - p1_outside (float): 1st percentile intensity in the area outside the container.
            - p50_outside (float): 50th percentile (median) intensity in the area outside the container.
            - p99_outside (float): 99th percentile intensity in the area outside the container.
            - m_outside (float): Mean intensity in the area outside the container.
            - std_outside (float): Standard deviation of intensity in the area outside the container.
    """
    # Load the image using PIL
    img = Image.open(os.path.join(path_to_tiff_stack, filename))
    # Convert the image to a numpy array
    img_array = np.array(img)
    mask_container, mask_outside, mask_core = get_masks_pre(img_array)

    # Stats for core
    p1_core = np.percentile(np.array(img_array[mask_core]), 1)
    p50_core = np.percentile(np.array(img_array[mask_core]), 50)
    p99_core = np.percentile(np.array(img_array[mask_core]), 99)
    m_core = np.mean(np.array(img_array[mask_core]))
    std_core = np.std(np.array(img_array[mask_core]))

    # Stats for plastic container
    p1_container = np.percentile(np.array(img_array[mask_container]), 1)
    p50_container = np.percentile(np.array(img_array[mask_container]), 50)
    p99_container = np.percentile(np.array(img_array[mask_container]), 99)
    m_container = np.mean(np.array(img_array[mask_container]))
    std_container = np.std(np.array(img_array[mask_container]))

    # Stats for area outside container
    p1_outside = np.percentile(np.array(img_array[mask_outside]), 1)
    p50_outside = np.percentile(np.array(img_array[mask_outside]), 50)
    p99_outside = np.percentile(np.array(img_array[mask_outside]), 99)
    m_outside = np.mean(np.array(img_array[mask_outside]))
    std_outside = np.std(np.array(img_array[mask_outside]))

    return (
        p1_core, p50_core, p99_core, m_core, std_core,
        p1_container, p50_container, p99_container, m_container, std_container,
        p1_outside, p50_outside, p99_outside, m_outside, std_outside
    )
