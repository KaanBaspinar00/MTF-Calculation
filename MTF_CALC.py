import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fft import fftfreq, fft

# Kaan Başpınar
# 2422855
# baspinarlee@gamil.com



# To see bad resolution, do this: blur = 201 ; width, height = 600, 600
# To see nice resolution, do this: blur = 21 ; width, height = 600, 600
# To see almost perfect resolution, do this: blur = 1 ; width, height = 600, 600

blur = 1 # note: this has to be odd number.

# Dimensions of image:
width, height = 600, 600


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


# Black Background
image = np.zeros((height, width, 3), dtype=np.uint8)

# Parameters for semi-circle
center = (int(width/2), int(height/2))
axes = (int(width/3), int(height/3))  # axes lengths
angle = 0  # Rotation angle
start_angle = 90  # for vertical semi-circle
end_angle = 270  # for vertical semi-circle
color = (255, 255, 255)

# Draw the semi-circle
cv2.ellipse(image, center, axes, angle, start_angle, end_angle, color, thickness=cv2.FILLED)

# Gaussian blur to whole image
blurred_image = cv2.GaussianBlur(image, (blur, blur), 0)

# Convert to grayscale
gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)

# Get a horizontal line
y_coordinate = int(height/2)
horizontal_line = gray_image[y_coordinate, :]
# center horizontal line from image

pixel_values = np.asarray(horizontal_line).flatten()[int(width/4):width - int(width/4)][::-1] / 255
# Normalize pixel values dividing by 255.

pixels_location = np.arange(0, int(width/2), 1)

# Display the image
cv2.imshow("Semi-circle Image", blurred_image)
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


