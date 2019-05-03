import numpy as np
import matplotlib.pyplot as plt
import json
from pprint import pprint
from numpy.linalg import norm
from statistics import median

NANOSECOND_TO_SECOND_FACTOR = 1000000000
lambda_M = 0.017  # Units: g


def normalize_times(data):
    return [(d[0] - data[0][0], d[1]) for d in data]


def compute_acceleration_magnitude(accelerometer_data):
    return normalize_times(
        [(int(timestamp) / NANOSECOND_TO_SECOND_FACTOR, norm((values["x"], values["y"], values["z"]))) for
         timestamp, values in accelerometer_data.items()])


def lambda_M_threshold(current_median, previous_median=None):
    if (previous_median is not None) and (abs(current_median - previous_median) < lambda_M):
        return previous_median
    else:
        return current_median


def object_median(func, object_list):
    return median([func(elem) for elem in object_list])


def compute_current_median(i, acceleration_magnitudes):
    return object_median(lambda pt: pt[1], acceleration_magnitudes[max(i - 2, 0):i + 1])


def get_previous_median(i, medians):
    return medians[i - 1][1] if 0 <= i - 1 < len(medians) else None


def compute_moving_median_filter(acceleration_magnitudes):
    filtered = []
    for i, point in enumerate(acceleration_magnitudes):
        filtered.append((point[0], lambda_M_threshold(compute_current_median(i, acceleration_magnitudes),
                                                      get_previous_median(i, filtered))))
    return filtered


if __name__ == '__main__':
    with open('pach-cardiac-Accelerometer-export.json') as f:
        accelerometer_data = json.load(f)

    Am = compute_acceleration_magnitude(accelerometer_data)
    AmL = compute_moving_median_filter(Am)


    plt.plot(*zip(*Am), '-.g')

    plt.plot(*zip(*AmL), '-r')

    plt.show()

    pprint(Am)

    pprint(AmL)
