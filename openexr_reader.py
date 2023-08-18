import OpenEXR
import Imath
import numpy as np
import pickle
# import matplotlib.pyplot as plt

depth_diff_list = []
for i in range(75000+1):
    # Open the EXR file
    # path = '/users/aarjun1/data/aarjun1/prj_depth/depth_images/{}.exr'.format(i)
    path = '/cifs/data/tserre_lrs/projects/prj_depth/dataset/blender_images_all/depth_maps/{}.exr'.format(i)
    file = OpenEXR.InputFile(path)

    # Get the size of the image
    dw = file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Read the three color channels as 32-bit floats
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    redstr = file.channel('R', FLOAT)
    greenstr = file.channel('G', FLOAT)
    bluestr = file.channel('B', FLOAT)

    # Convert the strings to numpy arrays and reshape to 2D
    red = np.frombuffer(redstr, dtype=np.float32)
    red.shape = (size[1], size[0])  # Numpy's "shape" is reversed
    # green = np.frombuffer(greenstr, dtype=np.float32)
    # green.shape = (size[1], size[0])
    # blue = np.frombuffer(bluestr, dtype=np.float32)
    # blue.shape = (size[1], size[0])

    print('###############################\n')
    print('i : ',i)
    print('red : ',red.shape)
    print('red max : ',red.max())
    print('red min : ',red.min())
    print('red diff : ',red.max()-red.min())
    print('###############################\n')

    depth_diff_list.append(red.max()-red.min())

depth_diff = np.array(depth_diff_list)

# Print Histogram
hist, bin_edges = np.histogram(depth_diff, bins=25)

print('Bin Range  : Count')
for start, end, count in zip(bin_edges[:-1], bin_edges[1:], hist):
    print('{:10.3f} - {:10.3f} : {}'.format(start, end, count))

print('#########################')
# Calculate the 45th and 55th percentiles
lower = np.percentile(depth_diff, 45)
upper = np.percentile(depth_diff, 55)

# Calculate the ranges
bottom_range = [np.min(depth_diff), lower]
top_range = [upper, np.max(depth_diff)]
print('bottom_range : ',bottom_range)
print('top_range : ',top_range)

target_dir = '/cifs/data/tserre_lrs/projects/prj_depth/dataset/blender_images_all'
depth_details = {'depth_diff_arr':depth_diff, 'lower_depth_range':bottom_range, 'upper_depth_range':top_range}
with open(target_dir + '/depth_details.pkl', 'wb') as f:
        pickle.dump(depth_details, f)

# # Plot Histogram
# plt.figure(figsize=(10,6))  # Set the size of the figure (optional)
# plt.hist(depth_diff_list, bins='auto', color='skyblue', edgecolor='black')  # 'auto' lets matplotlib decide the number of bins

# plt.title('Histogram of Terrain Height Differences')
# plt.xlabel('Height Difference')
# plt.ylabel('Frequency')

# plt.show()
# plt.savefig('/users/aarjun1/data/aarjun1/prj_depth/depth_diff.png')

# print('green : ',green.shape)
# print('green max : ',green.max())
# print('green min : ',green.min())

# print('blue : ',blue.shape)
# print('blue max : ',blue.max())
# print('blue min : ',blue.min())