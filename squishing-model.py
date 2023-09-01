#%% squishing model on unit circle

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Function to calculate the arclength of the quarter elliptical segment
def arclength_quarter_ellipse(x, flattened_width, a_ellipse, b_ellipse):
    derivative = b_ellipse * (-1/a_ellipse**2) * (x - flattened_width) / np.sqrt(1 - ((x - flattened_width) / a_ellipse)**2)
    return np.sqrt(1 + derivative**2)

# Radius of the initial quarter circle
initial_radius = 1
initial_x_values = np.linspace(0, initial_radius, 1000)
initial_y_values = np.sqrt(initial_radius**2 - initial_x_values**2)
initial_area_quarter_circle = 0.25 * np.pi * initial_radius**2

# Plot the initial quarter circle
# plt.plot(initial_x_values, initial_y_values, color="blue")
plt.fill_between(initial_x_values, 0, initial_y_values, alpha=0.3, color="blue")

# Squishing process
for t_squish in np.arange(0, 0.56, 0.05):
    # Height and width of the rectangle
    rectangle_height_squish = initial_radius - t_squish
    rectangle_width_squish = np.sqrt(initial_radius**2 - rectangle_height_squish**2)

    # Remaining area for the ellipse
    remaining_area_squish = initial_area_quarter_circle - (rectangle_height_squish * rectangle_width_squish)

    # Major axis (width) of the quarter ellipse
    a_ellipse = remaining_area_squish / (0.25 * np.pi * rectangle_height_squish)

    # Elliptical segment
    x_values_ellipse_squish = np.linspace(rectangle_width_squish, 25, 500000)
    y_values_ellipse_squish = rectangle_height_squish * np.sqrt(1 - ((x_values_ellipse_squish - rectangle_width_squish) / a_ellipse)**2)

    # Plot the rectangle (flattened region) with sequential color coding
    plt.plot([0, rectangle_width_squish, rectangle_width_squish], [rectangle_height_squish, rectangle_height_squish, 0], alpha=0.7, color=plt.cm.viridis(t_squish * 2))

    # Plot the quarter elliptical segment with sequential color coding
    plt.plot(x_values_ellipse_squish, y_values_ellipse_squish, alpha=0.7, color=plt.cm.viridis(t_squish * 2))
    plt.fill_between(x_values_ellipse_squish, 0, y_values_ellipse_squish, alpha=0.3, color=plt.cm.viridis(t_squish * 2))

plt.title("Transformation Visualization (t=0 to t=0.55)")
plt.xlim(0, 2.1)
plt.ylim(0, 1.1)
# plt.legend()
plt.show()


#%% deprecated 3-panel plot -- comparing ratio of curved to total length, curved and flat lengths vs squishing parameter

# Extending the simulation to t=3 and plotting the requested visualizations

# Storing the ratios and individual lengths
ratios_curved_to_total = []
curved_lengths = []
flat_lengths = []

# Squishing process up to t=3
for t_squish in np.arange(0, 0.56, 0.05):
    # Height and width of the rectangle
    rectangle_height_squish = initial_radius - t_squish
    rectangle_width_squish = np.sqrt(initial_radius**2 - rectangle_height_squish**2)

    # Remaining area for the ellipse
    remaining_area_squish = initial_area_quarter_circle - (rectangle_height_squish * rectangle_width_squish)

    # Major axis (width) of the quarter ellipse
    a_ellipse = remaining_area_squish / (0.25 * np.pi * rectangle_height_squish)

    # Elliptical segment
    x_values_ellipse_squish = np.linspace(rectangle_width_squish, rectangle_width_squish + a_ellipse, 1000)
    y_values_ellipse_squish = rectangle_height_squish * np.sqrt(1 - ((x_values_ellipse_squish - rectangle_width_squish) / a_ellipse)**2)

    # Calculate the arclength of the quarter elliptical segment
    if t_squish == 0:
        curved_arclength = initial_radius * np.pi / 2  # Arclength of the initial quarter circle
    else:
        curved_arclength, _ = quad(arclength_quarter_ellipse, rectangle_width_squish, rectangle_width_squish + a_ellipse, args=(rectangle_width_squish, a_ellipse, rectangle_height_squish))

    # Calculate the ratio of curved length to total length (rectangle width + curved length)
    total_length = rectangle_width_squish + curved_arclength
    ratio = curved_arclength / total_length
    ratios_curved_to_total.append(ratio)
    curved_lengths.append(curved_arclength)
    flat_lengths.append(rectangle_width_squish)

# Plotting the ratio of the curved length to the total length
plt.figure(figsize=[15, 4])
plt.subplot(1, 4, 1)
plt.plot(np.arange(0, 0.56, 0.05), ratios_curved_to_total, marker='o')
plt.title("Ratio of Curved Length to Total Length")
plt.xlabel("t (Squishing Parameter)")
plt.ylabel("Ratio")
plt.grid(True)

# Plotting the individual lengths of the curved and flat portions
plt.subplot(1, 4, 2)
plt.plot(np.arange(0, 0.56, 0.05), curved_lengths, marker='o', label="Curved Length")
plt.plot(np.arange(0, 0.56, 0.05), flat_lengths, marker='o', label="Flat Length")
plt.title("Curved and Flat Lengths")
plt.xlabel("t (Squishing Parameter)")
plt.ylabel("Length")
plt.legend()
plt.grid(True)

# Visualization of the transformation process
plt.subplot(1, 4, 3)
plt.plot(initial_x_values, initial_y_values, color="blue")
plt.fill_between(initial_x_values, 0, initial_y_values, alpha=0.3, color="blue")
for t_squish in np.arange(0, 0.56, 0.05):
    rectangle_height_squish = initial_radius - t_squish
    rectangle_width_squish = np.sqrt(initial_radius**2 - rectangle_height_squish**2)
    remaining_area_squish = initial_area_quarter_circle - (rectangle_height_squish * rectangle_width_squish)
    a_ellipse = remaining_area_squish / (0.25 * np.pi * rectangle_height_squish)
    x_values_ellipse_squish = np.linspace(rectangle_width_squish, rectangle_width_squish + a_ellipse, 1000)
    y_values_ellipse_squish = rectangle_height_squish * np.sqrt(1 - ((x_values_ellipse_squish - rectangle_width_squish) / a_ellipse)**2)
    plt.plot([0, rectangle_width_squish, rectangle_width_squish], [rectangle_height_squish, rectangle_height_squish, 0], alpha=0.7, color=plt.cm.viridis(t_squish * 3))
    plt.plot(x_values_ellipse_squish, y_values_ellipse_squish, alpha=0.7, color=plt.cm.viridis(t_squish * 3))
    plt.xlim(0, 2.1)
    plt.ylim(0, 2.1)


plt.title("Transformation Visualization (t=0 to t=0.55)")

plt.tight_layout()
plt.show()



#%% aspect ratio

import numpy as np
import matplotlib.pyplot as plt

# Radius of the initial quarter circle
initial_radius = 1
initial_area_quarter_circle = 0.25 * np.pi * initial_radius**2

# Storing the aspect ratios for different t values
aspect_ratios = []

# Squishing process up to t=3
for t_squish in np.arange(0, 0.56, 0.05):
    rectangle_height_squish = max(initial_radius - t_squish, 0)
    rectangle_width_squish = np.sqrt(max(initial_radius**2 - rectangle_height_squish**2, 0))
    remaining_area_squish = max(initial_area_quarter_circle - (rectangle_height_squish * rectangle_width_squish), 0)
    a_ellipse = 0 if remaining_area_squish == 0 else remaining_area_squish / (0.25 * np.pi * rectangle_height_squish)

    # Total width of the shape (rectangle width + quarter ellipse width)
    total_width = rectangle_width_squish + a_ellipse

    # Aspect ratio (width to height)
    aspect_ratio = total_width / rectangle_height_squish if rectangle_height_squish > 0 else 0
    aspect_ratios.append(aspect_ratio)

# Plotting the aspect ratio vs t
plt.plot(np.arange(0, 0.56, 0.05), aspect_ratios, marker='o')
plt.title("Aspect Ratio (Width to Height) vs Squishing Parameter (t)")
plt.xlabel("Squishing Parameter (t)")
plt.ylabel("Aspect Ratio (Width / Height)")
plt.grid(True)
plt.show()



#%% deprecated curvature (curved part only)

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Function to calculate the curvature of the quarter elliptical segment
def curvature_quarter_ellipse(theta, a_ellipse, b_ellipse):
    return (a_ellipse * b_ellipse) / ((a_ellipse**2 * np.sin(theta)**2 + b_ellipse**2 * np.cos(theta)**2)**(3/2))

# Function to calculate arclength of the quarter ellipse given curvature
def arclength_given_curvature(theta, a_ellipse, b_ellipse):
    curvature = curvature_quarter_ellipse(theta, a_ellipse, b_ellipse)
    return np.sqrt(1 + curvature**2)

# Radius of the initial quarter circle
initial_radius = 1
initial_area_quarter_circle = 0.25 * np.pi * initial_radius**2

# Storing the curvature as a function of normalized arclength for different t values
curvature_vs_arclength_data = []

# Squishing process up to t=3
for t_squish in np.arange(0, 0.56, 0.05):
    rectangle_height_squish = max(initial_radius - t_squish, 0)
    rectangle_width_squish = np.sqrt(max(initial_radius**2 - rectangle_height_squish**2, 0))
    remaining_area_squish = max(initial_area_quarter_circle - (rectangle_height_squish * rectangle_width_squish), 0)
    a_ellipse = 0 if remaining_area_squish == 0 else remaining_area_squish / (0.25 * np.pi * rectangle_height_squish)
    b_ellipse = rectangle_height_squish
    theta_values = np.linspace(0, np.pi/2, 1000)
    curvature_values = curvature_quarter_ellipse(theta_values, a_ellipse, b_ellipse)
    arclength_values = [quad(arclength_given_curvature, 0, theta, args=(a_ellipse, b_ellipse))[0] for theta in theta_values]
    max_arclength = max(arclength_values, default=1)
    arclength_values_normalized = [arclength / max_arclength for arclength in arclength_values]
    curvature_vs_arclength_data.append((arclength_values_normalized, curvature_values))

# Plotting the curvature as a function of normalized arclength for different t values
plt.figure(figsize=[12, 6])
for i, t_squish in enumerate(np.arange(0, 0.56, 0.05)):
    plt.plot(curvature_vs_arclength_data[i][0], curvature_vs_arclength_data[i][1], color=plt.cm.viridis(i / len(curvature_vs_arclength_data)))
plt.axhline(y=1/initial_radius, color='r', linestyle='--', label=f'Curvature of Initial Circle (t=0)')
plt.title("Curvature vs Normalized Arclength for Different Squishing Parameters")
plt.xlabel("Normalized Arclength")
plt.ylabel("Curvature")
plt.legend()
plt.grid(True)
plt.show()

#%% new curvature (with flat part)

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Function to calculate the arclength of the quarter elliptical segment
def arclength_quarter_ellipse(x, flattened_width, a_ellipse, b_ellipse):
    derivative = b_ellipse * (-1/a_ellipse**2) * (x - flattened_width) / np.sqrt(1 - ((x - flattened_width) / a_ellipse)**2)
    return np.sqrt(1 + derivative**2)

# Radius of the initial quarter circle
initial_radius = 1
initial_area_quarter_circle = 0.25 * np.pi * initial_radius**2

# Storing the curvature as a function of normalized arclength for different t values
curvature_vs_arclength_data = []

# Squishing process up to t=3
for t_squish in np.arange(0, 0.56, 0.05):
    rectangle_height_squish = initial_radius - t_squish
    rectangle_width_squish = np.sqrt(initial_radius**2 - rectangle_height_squish**2)
    remaining_area_squish = initial_area_quarter_circle - (rectangle_height_squish * rectangle_width_squish)
    a_ellipse = remaining_area_squish / (0.25 * np.pi * rectangle_height_squish)
    
    # Elliptical segment
    x_values_ellipse_squish = np.linspace(rectangle_width_squish, rectangle_width_squish + a_ellipse, 1000)
    arclength_values_curved_part, _ = quad(arclength_quarter_ellipse, rectangle_width_squish, rectangle_width_squish + a_ellipse, args=(rectangle_width_squish, a_ellipse, rectangle_height_squish))
    total_arclength = rectangle_width_squish + arclength_values_curved_part
    
    # Normalized arclength for curved and flat parts
    arclength_values_normalized_curved = np.linspace(0, arclength_values_curved_part / total_arclength, 1000)
    arclength_values_normalized_flat = np.linspace(arclength_values_curved_part / total_arclength, 1, 1000)

    # Curvature values (calculated values for curved part, 0 for flat part)
    curvature_values_curved = curvature_quarter_ellipse(arclength_values_normalized_curved * np.pi/2, a_ellipse, rectangle_height_squish)
    curvature_values_flat = np.zeros_like(arclength_values_normalized_flat)
    
    # Concatenating curved and flat parts
    arclength_values_normalized_total = np.concatenate((arclength_values_normalized_curved, arclength_values_normalized_flat))
    curvature_values_total = np.concatenate((curvature_values_curved, curvature_values_flat))

    # Store the data for this t value
    curvature_vs_arclength_data.append((arclength_values_normalized_total, curvature_values_total))

# Plotting the curvature as a function of normalized arclength for different t values
plt.figure(figsize=[12, 6])
for i, t_squish in enumerate(np.arange(0, 0.56, 0.05)):
    plt.plot(curvature_vs_arclength_data[i][0], curvature_vs_arclength_data[i][1], label=f't={t_squish:.2f}', color=plt.cm.viridis(i / len(curvature_vs_arclength_data)))
plt.title("Curvature vs Normalized Arclength for Entire Shape (Including Flat Part)")
plt.xlabel("Normalized Arclength")
plt.ylabel("Curvature")
plt.legend()
plt.grid(True)
plt.show()

#%% immediately pre-flat part curvature value

# Extracting the final curvature value before it becomes 0 for each t value
final_curvature_values = [curvature_values[curvature_values > 0][-1] if curvature_values[curvature_values > 0].size > 0 else 0 for _, curvature_values in curvature_vs_arclength_data]
t_values = np.arange(0, 0.56, 0.05)

# Plotting the final curvature value vs t
plt.figure(figsize=[10, 5])
plt.plot(t_values, final_curvature_values, marker='o')
plt.title("Final Curvature Value Before 0 vs Squishing Parameter (t)")
plt.xlabel("Squishing Parameter (t)")
plt.ylabel("Final Curvature Value Before 0")
plt.grid(True)
plt.show()


#%% average curvature and variance of curvature for only curved parts

# Storing the average curvature and variance for each t value
average_curvatures = []
curvature_variances = []

# Extracting the curvature values for the curved segment only (excluding zeros)
for t_data in curvature_vs_arclength_data:
    curvature_values_curved = t_data[1][t_data[1] > 0]
    average_curvatures.append(np.mean(curvature_values_curved))
    curvature_variances.append(np.var(curvature_values_curved))

# Plotting the average curvature and variance for each t value
t_values = np.arange(0, 0.56, 0.05)
plt.figure(figsize=[12, 6])
plt.plot(t_values, average_curvatures, label='Average Curvature', color='b')
plt.plot(t_values, curvature_variances, label='Curvature Variance', color='r')
plt.title("Average Curvature and Curvature Variance vs Squishing Parameter (t)")
plt.xlabel("Squishing Parameter (t)")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()


#%% % of the arclength that is flat (redundant with deprecated 3-panel above)

# Calculate the percentage of the arclength that is flat for each t value
percentage_flat_arclength = []

for arclength_values, curvature_values in curvature_vs_arclength_data:
    # Finding the point where the curvature becomes zero
    index_curvature_zero = np.argmax(curvature_values == 0)
    
    # Percentage of the arclength that is flat
    percentage_flat = (1 - arclength_values[index_curvature_zero]) * 100
    percentage_flat_arclength.append(percentage_flat)

# Plotting the percentage of the arclength that is flat vs t
t_values = np.arange(0, 0.56, 0.05)
plt.plot(t_values, percentage_flat_arclength, marker='o')
plt.title("Percentage of Normalized Arclength with Curvature = 0 vs t")
plt.xlabel("t")
plt.ylabel("Percentage of Flat Arclength")
plt.grid(True)
plt.show()



