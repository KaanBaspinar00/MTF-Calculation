import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fft import fftfreq, fft

# Kaan Başpınar
# 2422855
# baspinarlee@gamil.com
# for github page: https://github.com/KaanBaspinar00/MTF-Calculation

# Image path
image_path = "deneme2.png"

# Read the image
image = cv2.imread(image_path)

# Resize the image
resize_x, resize_y = 500, 500
image = cv2.resize(image, [resize_x, resize_y])

# Dimensions of image:
width, height = image.shape[0], image.shape[1]


# Apply fft
def fourier_analysis(line_spread_x, line_spread_y):
    """
    Perform Fourier analysis on the data.

    Computes the Fourier frequencies, performs Fourier transform,
    and normalizes the result to its maximum value.
    """

    sample_rate = len(line_spread_x) / (max(line_spread_x) - min(line_spread_x))
    freq = fftfreq(len(line_spread_x), 1 / sample_rate)
    data = line_spread_y
    fourier_transformed = np.array(fft(np.array(data))[:])
    fourier_transformed_real = np.abs(fourier_transformed / np.max(fourier_transformed))

    # Filtered data where x is greater than 0
    filtered_x = np.array([x for x in freq if x > 0])
    filtered_y = np.array([fourier_transformed_real[list(freq).index(x)] for x in filtered_x])

    return freq, filtered_x, filtered_y, fourier_transformed


# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Get a horizontal line
y_coordinate = int(height/2)
horizontal_line = gray_image[y_coordinate, :]
# center horizontal line from image

pixel_values = np.asarray(horizontal_line).flatten()[::-1]/255
# Normalize pixel values dividing by 255.


gray_image_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
pixels_location = np.arange(0, width, 1)

gray_image = cv2.line(gray_image_bgr, (0, y_coordinate), (width, y_coordinate), (0, 0, 255), thickness=2)
# Display the image
cv2.imshow("Image", gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Compute the derivative using finite differences
dy_dx = np.gradient(pixel_values, 1)

# Create a figure and subplots
fig, axs = plt.subplots(3, 1)

# Plot pixel values
axs[0].plot(pixels_location, pixel_values, "-r")
axs[0].set_xlabel("pixel #")
axs[0].set_ylabel("pixel value")
axs[0].set_title("Pixels values for a line")
axs[0].grid()

# Plot derivative of pixel values
axs[1].plot(pixels_location, dy_dx, "-b")
axs[1].set_xlabel("pixel #")
axs[1].set_ylabel("dy_dx")
axs[1].set_title("Line Spread Function")
axs[1].grid()

# Calculate Fourier transform of MTF
X, Y = fourier_analysis(pixels_location, dy_dx)[1], fourier_analysis(pixels_location, dy_dx)[2]

# Plot MTF
axs[2].plot(X, Y, "-k")
axs[2].set_xlabel("pixel #")
axs[2].set_ylabel("FFT of dy_dx")
axs[2].set_title("MTF")
axs[2].grid()

plt.tight_layout()  # Adjust subplot parameters to give specified padding
plt.show()
