import numpy as np
import cv2
import math

cap = cv2.VideoCapture("test_frames/360_0087.MP4")

success, frame = cap.read()
# cv2.imshow("video", frame)
# key = cv2.waitKey(0)
cv2.imwrite("frame.png",frame)
# cv2.imshow("video", frame)
height, width, rgb = frame.shape

# we assume that the FOV for one of the fisheye lenses is 180 deg for simplicity
fov = 180
fov = np.deg2rad(fov)

width_cutoff = width // 2 # should be 1280
circle_radius = width_cutoff // 2 # should be 640
left_lens = frame[:, :width_cutoff, :]
right_lens = frame[:, width_cutoff:, :]

# get rid of vignette
# calculate the vignette values at different radius
p1 = -7.5625e-17
p2 = 1.9589e-13
p3 = -1.8547e-10
p4 = 6.1997e-8
p5 = -6.9432e-5
p6 = 0.9976

pos_distance = np.arange(circle_radius)
brightness = []
inv_brightness = []
inv_brightness_norm = []
for x in pos_distance:
    x_scale = x
    p = p1*(x_scale**5) + p2*(x_scale**4) + p3*(x_scale**3) + p4*(x_scale**2) + p5*(x_scale) + p6
    brightness.append((p*255))
    inv_brightness.append(255-(p*255))
    inv_brightness_norm.append(1-p)

# generate a 2d matrix of vignette model
vignette_image = np.zeros((height, width_cutoff))
inv_vignette_image = np.zeros((height, width_cutoff))
inv_brightness_norm_image = np.zeros((height, width_cutoff))
for row_i in np.arange(width_cutoff):
    for column_i in np.arange(height):
        distance = math.dist([column_i, row_i], [circle_radius, circle_radius])
        dist_index = np.argmin(np.abs(np.array(pos_distance)-distance))
        vignette_image[row_i, column_i] = brightness[dist_index]
        inv_vignette_image[row_i, column_i] = inv_brightness[dist_index]
        inv_brightness_norm_image[row_i, column_i] = inv_brightness_norm[dist_index]

# show vignette
vignette_image = cv2.normalize(vignette_image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
vignette_image = cv2.cvtColor(vignette_image, cv2.COLOR_GRAY2BGR)
# cv2.imshow("video", vignette_image)
# key = cv2.waitKey(0)
cv2.imwrite("vignette_model.png",vignette_image)

# show inverse vignette
inv_vignette_image = cv2.normalize(inv_vignette_image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
inv_vignette_image = cv2.cvtColor(inv_vignette_image, cv2.COLOR_GRAY2BGR)
# cv2.imshow("video", inv_vignette_image)
# key = cv2.waitKey(0)
cv2.imwrite("inv_vignette_model.png",inv_vignette_image)

# remove the vignetting template from the image
# https://stackoverflow.com/questions/74786867/subtract-vignetting-template-from-image-in-opencv-python (used this source for inspiration)
vignette_image = cv2.cvtColor(vignette_image, cv2.COLOR_BGR2GRAY)
vignette_image = cv2.medianBlur(vignette_image, 15)
vignette_norm = vignette_image.astype(np.float32) / 255
vignette_norm = cv2.GaussianBlur(vignette_norm, (51, 51), 30)
vig_mean_val = cv2.mean(vignette_norm)[0]
inv_vig_norm = vig_mean_val / vignette_norm
inv_vig_norm = cv2.cvtColor(inv_vig_norm, cv2.COLOR_GRAY2BGR) 

left_lens_vignette = cv2.multiply(left_lens, inv_vig_norm, dtype=cv2.CV_8U) 
right_lens_vignette = cv2.multiply(right_lens, inv_vig_norm, dtype=cv2.CV_8U) 
left_right_fish_no_vignette = np.hstack([left_lens_vignette[:, :, :], right_lens_vignette[:, :, :]])
cv2.imwrite("fish_no_vignette.png",left_right_fish_no_vignette)

# equirectangular projection
x_prime, y_prime = np.meshgrid(np.arange(width_cutoff, dtype=np.float32), np.arange(width_cutoff, dtype=np.float32))

x_prime = x_prime - circle_radius
y_prime = circle_radius - y_prime

theta_s = x_prime * fov / width_cutoff - np.deg2rad(0.5)
phi_s = y_prime * fov / height - np.deg2rad(0.5)

x = np.sin(theta_s) * np.cos(phi_s)
y = np.cos(theta_s) * np.cos(phi_s)
z = np.sin(phi_s)

theta = np.arctan2(z, x)
rho = height * np.arctan2(np.sqrt(x**2 + z**2), y) / fov

y_dprime = 0.5 * height - rho * np.sin(theta)
x_dprime = 0.5 * width_cutoff + rho * np.cos(theta)

left_equi = cv2.remap(left_lens, x_dprime, y_dprime, cv2.INTER_LINEAR)
right_equi = cv2.remap(right_lens, x_dprime, y_dprime, cv2.INTER_LINEAR)
left_right_with_overlap = np.hstack([left_equi[:, :, :], right_equi[:, :, :]])
cv2.imwrite("no_vignette_overlap.png",left_right_with_overlap)

left_equi_vignette = cv2.remap(left_lens_vignette, x_dprime, y_dprime, cv2.INTER_LINEAR)
right_equi_vignette = cv2.remap(right_lens_vignette, x_dprime, y_dprime, cv2.INTER_LINEAR)

# overlapping region is first 75 rows on both sides
overlapping_region = np.hstack([left_equi[:, 0:75, :], np.zeros((1280, width_cutoff-75, 3)), right_equi[:, 0:75, :], np.zeros((1280, width_cutoff-75, 3))])

full_equi = np.hstack([left_equi[:, 75:width_cutoff, :], right_equi[:, 75:width_cutoff, :]])
cv2.imwrite("vignette_final.png",full_equi)
full_equi_vignette = np.hstack([left_equi_vignette[:, 75:width_cutoff, :], right_equi_vignette[:, 75:width_cutoff, :]])
cv2.imwrite("final.png",full_equi_vignette)

cv2.imshow("video", full_equi)
key = cv2.waitKey(0)

cv2.imshow("video", full_equi_vignette)
key = cv2.waitKey(0)