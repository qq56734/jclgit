import numpy as np

import matplotlib.pyplot as plt



def get_command(direction0, loc1, loc2):
    direction = loc2 - loc1

    distance = np.linalg.norm(direction)

    cosangle = direction.dot(direction0) / (np.linalg.norm(direction) * np.linalg.norm(direction0))

    print(cosangle)

    angle = np.arccos(cosangle)*360/2/np.pi

    if direction0[0]*direction[1] - direction[0]*direction0[1] < 0:
        angle = - angle

    return angle, distance





current_pos0 = np.array([1156, 940])
current_pos1 = np.array([1000, 940])

direction0 = current_pos1 - current_pos0


points = np.array([current_pos0, [200, 800], [200,200], [1120, 500]])

for i in range(1,len(points) - 1):
    print(get_command(points[i]- points[i-1], points[i], points[i + 1]))

#
#

plt.plot(points[:,0], points[:, 1], '.')
plt.plot(current_pos1[0],current_pos1[1], '.')
plt.show()
#
