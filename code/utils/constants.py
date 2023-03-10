import numpy as np


def column_name_to_ind(s: str) -> int:
    if s[0] == 'x':
        return 2 * int(s[2:])
    if s[0] == 'y':
        return 2 * int(s[2:]) + 1

    return -1


E = 3
THETA = 4

TARGET_COLUMNS = ['acc_z', 'acc_y', 'acc_x', 'gyr_z', 'gyr_y', 'gyr_x']
TARGET_SIZE = len(TARGET_COLUMNS)

KEYPOINTS_CNT = 68
VIDEO_COLUMNS = np.array([
    'x_0', 'y_0', 'x_8', 'y_8', 'x_10', 'y_10', 'x_51', 'y_51'
])
video_columns_indices = list(map(column_name_to_ind, VIDEO_COLUMNS))

EXCESS_VIDEO_COLS = list(range(KEYPOINTS_CNT * 2))
for ind in video_columns_indices:
    EXCESS_VIDEO_COLS.remove(ind)

SUBDIRS = (
    'round_and_round',
    'chaotic_1',
    'chaotic_2',
    'chaotic_3',
    'cyclic_1',
    'cyclic_2'
)
PRED_METHODS = ('ccm', 'pls', 'cca', 'naive')