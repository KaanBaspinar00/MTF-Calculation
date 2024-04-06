import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fft import fftfreq, fft

# Kaan Başpınar
# 2422855
# baspinarlee@gamil.com
# for github page: https://github.com/KaanBaspinar00/MTF-Calculation

# Example usage:
initial_angle, final_angle, number_of_steps = 0, 2*np.pi, 1000

theta = np.linspace(initial_angle, final_angle, number_of_steps)
radius = 150

resize_x, resize_y = 500, 500
# Upload image
image_path = "deneme6.png"

# Read the image
image = cv2.imread(image_path)

image = cv2.resize(image, [resize_x, resize_y])

# Dimensions of image:
width, height = image.shape[0], image.shape[1]


def create_circle_mask(center_x, center_y, radius, theta):
    x = []
    y = []

    for i in theta:
        x.append(center_x + radius*np.cos(i))
        y.append(center_y - radius*np.sin(i))
    return x, y


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


# Get circular curve
x_coordinate = create_circle_mask(int(width/2), int(height/2), radius, theta)[0]
y_coordinate = create_circle_mask(int(width/2), int(height/2), radius, theta)[1]

# center horizontal line from image
pixel_values = []
for x, y in zip(x_coordinate, y_coordinate):
    x = int(x)
    y = int(y)
    pixel_value = gray_image[y, x]
    pixel_values.append(pixel_value)

# pixel_values = np.asarray(horizontal_line).flatten()[::-1]/255 #[int(width/4):width - int(width/4)][::-1] / 255
# Normalize pixel values dividing by 255.
# Draw a red line on the image at the specified coordinates
# Convert the grayscale image to BGR format for displaying the red line
gray_image_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

# Draw a red line on the image at the specified coordinates
for x, y in zip(x_coordinate, y_coordinate):
    x = int(x)
    y = int(y)
    cv2.line(gray_image_bgr, (x, y), (x, y), (0, 0, 255), thickness=2)  # Draw a red point (pixel) at the coordinate

# Display the image
cv2.imshow("Semi-circle Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Display the image with the red line
cv2.imshow("Image with Red Line", gray_image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

pixels_location = theta


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
