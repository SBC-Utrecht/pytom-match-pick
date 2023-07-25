import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom


wedge_angle = angle1 = np.deg2rad(20)
angle2 = np.deg2rad(85)
print(np.tan(angle1), np.tan(angle2))
size_x, size_y, size_z = (462, 478, 250)
size = 478

y = np.abs(np.arange(-size_y // 2 + size_y % 2, size_y // 2 + size_y % 2, 1.))[:, np.newaxis]
z = np.arange(0, size_z // 2 + 1, 1.)

wedge = np.zeros((size_y, size_z // 2 + 1))

with np.errstate(all='ignore'):
    wedge[y - np.tan(wedge_angle) * z > 0] = 1

wedge1 = np.zeros((size_y, size_z // 2 + 1))
wedge2 = np.zeros((size_y, size_z // 2 + 1))

with np.errstate(all='ignore'):
    wedge1[np.tan(angle1) < y / z] = 1
    wedge2[np.tan(angle2) < y / z] = 1

full_wedge = np.zeros((size_y, size_z // 2 + 1))
full_wedge[0:size_y // 2, :] = wedge2[0:size_y // 2, :]
full_wedge[size_y // 2:, :] = wedge1[size_y // 2:, :]

y = np.abs(np.arange(-size // 2 + size % 2, size // 2 + size % 2, 1.))[:, np.newaxis]
z = np.arange(0, size // 2 + 1, 1.)

edge_pixels = 0.5
new_wedge = y - np.tan(wedge_angle) * z
new_wedge[new_wedge > edge_pixels] = edge_pixels
new_wedge[new_wedge < -edge_pixels] = -edge_pixels
new_wedge = (new_wedge - new_wedge.min()) / (new_wedge.max() - new_wedge.min())

out_wedge = np.zeros((size_y, size_z // 2 + 1))
zoom(new_wedge, [size_y / size, (size_z // 2 + 1) / (size // 2 + 1)], output=out_wedge)

wedge_3d = np.tile(out_wedge, (462, 1, 1))
print(wedge_3d.shape)

fig, ax = plt.subplots(1, 4)
ax[0].imshow(np.fft.ifftshift(wedge, axes=0))
ax[1].imshow(np.fft.ifftshift(y - np.tan(wedge_angle) * z, axes=0))
ax[2].imshow(np.fft.ifftshift(out_wedge, axes=0))
ax[3].imshow(np.fft.ifftshift(full_wedge, axes=0))
plt.show()

