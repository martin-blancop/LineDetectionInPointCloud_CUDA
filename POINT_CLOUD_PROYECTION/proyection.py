import pdal
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

json = '''
[
    {
        "type": "readers.las",
        "filename": "input.laz"
    },
    {
        "type": "filters.range",
        "limits": "Classification[6:6]"
    },
    {
        "type":"filters.outlier",
        "method":"statistical",
        "mean_k":12,
        "multiplier":2.2
    },
    {
        "type": "writers.las",
        "filename": "output.las"
    }
]
'''

pipeline = pdal.Pipeline(json)
pipeline.execute()

# Retrieve the filtered point cloud data
point_cloud = pipeline.arrays[0]

# Extract X and Y coordinates
x_values = point_cloud["X"]
y_values = point_cloud["Y"]
red_values = point_cloud["Red"]
green_values = point_cloud["Green"]
blue_values = point_cloud["Blue"]

# Normalize color values to the range [0, 255]
max_value = np.max([np.max(red_values), np.max(green_values), np.max(blue_values)])
red_values = (red_values / max_value * 255).astype(np.uint8)
green_values = (green_values / max_value * 255).astype(np.uint8)
blue_values = (blue_values / max_value * 255).astype(np.uint8)

# Determine the grid size based on the range of X and Y values
x_min, x_max = np.min(x_values), np.max(x_values)
y_min, y_max = np.min(y_values), np.max(y_values)
grid_size = 1.0  # Adjust as needed

grid_values = [0.1, 0.2, 0.5, 1.0, 2.0]

for i in range(len(grid_values)):

    x_grid = np.arange(x_min, x_max + grid_values[i], grid_values[i])
    y_grid = np.arange(y_min, y_max + grid_values[i], grid_values[i])

    # Create an adjacency matrix and color array
    classification_array = np.zeros((len(y_grid), len(x_grid)), dtype=int)
    color_array = np.zeros((len(y_grid), len(x_grid), 3), dtype=np.uint8)  # RGB color

    # Iterate over the points and mark the corresponding grid cells as 1
    for x, y, red, green, blue in zip(x_values, y_values, red_values, green_values, blue_values):
        x_index = np.where(x_grid <= x)[0][-1]
        y_index = np.where(y_grid <= y)[0][-1]
        classification_array[y_index, x_index] = 1
        color_array[y_index, x_index] = (red, green, blue)

    classification_array = np.fliplr(np.flip(classification_array))
    color_array = np.fliplr(np.flip(color_array))

    # Convert adjacency matrix to image
    image = Image.fromarray((classification_array * 255).astype(np.uint8))

    # Save the image
    image.save("CLAS_" + str(grid_values[i]) + ".png")

    # Save the color array
    color_image = Image.fromarray(color_array, mode="RGB")
    color_image.save("COLOR_" + str(grid_values[i]) + ".png")