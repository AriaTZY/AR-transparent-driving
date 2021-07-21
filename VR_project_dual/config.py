import os

# # 相机自身安装俯仰角
# camera_fixed_pitch = 3.5
# # 相机固定高度（mm）
# camera_fixed_height = 1600
# # 左边摄像机相对中间摄像机的偏移角度
# left_camera_yaw_gap = -50.
# # 右边摄像机相对中间摄像机的偏移角度
# right_camera_yaw_gap = -50.
# # 初始观察角度
# look_init_yaw = 40
# look_init_pitch = 10

camera_order = [1, 2]
com_name = 'com4'

# 相机自身安装俯仰角
camera_fixed_pitch = 3.5
# 相机固定高度（mm）
camera_fixed_height = 1600
# 左边摄像机相对中间摄像机的偏移角度
left_camera_yaw_gap = -20.
# 右边摄像机相对中间摄像机的偏移角度
right_camera_yaw_gap = -50.
# 初始观察角度
look_init_yaw = 20
look_init_pitch = 10

# 左右观察角度限幅
max_look_left = 51
max_look_right = -51

# 图片大小
picture_size = (360, 640)