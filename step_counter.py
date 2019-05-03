import numpy as np
import matplotlib.pyplot as plt
import json
from pprint import pprint
from numpy.linalg import norm
from statistics import median
from statistics import mean

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


def object_mean(func, object_list):
    return mean([func(elem) for elem in object_list])


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


def compute_moving_average_filter(moving_median_filter):
    return [(point[0], object_mean(lambda pt: pt[1], moving_median_filter[max(0, i - 7):i + 1]))
            for i, point in enumerate(moving_median_filter)]


def compute_threshold(data):
    return [(point[0], max(1.033, data[i - 4 if i - 4 > 0 else i][1])) for i, point in enumerate(data)]


def count_steps(data, lambdaD):
    steps = 0
    peaks = 0
    armed = False
    falling = False

    for i, point in enumerate(data):
        if point[1] > lambdaD[i][1] and not armed:
            peaks += 1
            armed = True
        elif point[1] < lambdaD[i][1] and armed:
            falling = True
            steps += 1

            armed = False
            falling = False

    return steps


if __name__ == '__main__':
    with open('pach-cardiac-Accelerometer-export.json') as f:
        accelerometer_data = json.load(f)

    Am = compute_acceleration_magnitude(accelerometer_data)
    AmL = compute_moving_median_filter(Am)

    moving_avg_filter = compute_moving_average_filter(AmL)

    AmH = [(pt[0], pt[1] - moving_avg_filter[i][1]) for i, pt in enumerate(AmL)]

    threshold = compute_threshold(AmL)
    thresholdavg = compute_threshold(moving_avg_filter)

    # plt.plot(*zip(*Am))
    # plt.plot(*zip(*AmL))
    # plt.plot(*zip(*moving_avg_filter))
    # plt.plot(*zip(*threshold), "-.")
    # plt.plot(*zip(*AmH))

    # plt.show()

    pprint(Am)
    pprint(AmL)

    print("Number of Steps from AmL: {}".format(count_steps(AmL, threshold)))
    print("Number of Steps from Mean Filter: {}".format(count_steps(moving_avg_filter, threshold)))
    # print("Number of Steps from Mean Filter: {}".format(count_steps(AmH, threshold)))
