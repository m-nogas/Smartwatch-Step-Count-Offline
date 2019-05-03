import numpy as np
import matplotlib.pyplot as plt
import json
from pprint import pprint
from numpy.linalg import norm

NANOSECOND_TO_SECOND_FACTOR = 1000000000
lambda_M = 0.017  # Units: g


def normalize_times(data):
    return [(d[0] - data[0][0], d[1]) for d in data]


def compute_acceleration_magnitude(accelerometer_data):
    return normalize_times(
        [(int(timestamp) / NANOSECOND_TO_SECOND_FACTOR, norm((values["x"], values["y"], values["z"]))) for
         timestamp, values in accelerometer_data.items()])


if __name__ == '__main__':
    with open('pach-cardiac-Accelerometer-export.json') as f:
        accelerometer_data = json.load(f)

    Am = compute_acceleration_magnitude(accelerometer_data)

    ts, vals = zip(*Am)

    plt.plot(ts, vals, '-g')
    plt.show()

    pprint(Am)
