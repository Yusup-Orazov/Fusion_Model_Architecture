import laspy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def plot_las_file(file_path):
    with laspy.open(file_path) as las_file:
        point_data = las_file.read()
        x = point_data.x
        y = point_data.y
        z = point_data.z
        classifications = point_data.classification

        # Plotting
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(x, y, z, c=classifications, cmap='viridis', marker='.')
        ax.set_title(f"3D Plot of {os.path.basename(file_path)}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.colorbar(scatter, ax=ax, label='Classifications')
        plt.show()

        # Print unique classes
        unique_classes = set(classifications)
        print(f"File: {os.path.basename(file_path)} - Unique Classes: {unique_classes}")

def main(directory):
    for file_name in os.listdir(directory):
        if file_name.endswith('.las'):
            file_path = os.path.join(directory, file_name)
            plot_las_file(file_path)

if __name__ == '__main__':
    directory_path = '70k_las_plz_str_100m'
    main(directory_path)
