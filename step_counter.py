import matplotlib.pyplot as plt
import json
from pprint import pprint
from numpy.linalg import norm
from statistics import median
from statistics import mean
import numpy as np

NANOSECOND_TO_SECOND_FACTOR = 1000000000
lambda_M = 0.017  # Units: g

A_xmin = 0.25  # Units: g
A_ymin = 0.15  # Units: g
A_zmin = -0.36  # Units: g
A_zmax = 0.80  # Units: g
Acc_max = 2.50  # Units: g
Acc_min = 1.04  # Units: g
T_stmax = 1.50  # Units: Seconds
T_stmin = 0.30  # Units: Seconds
T = 0.12  # Units: Seconds
St_min = 6  # Units: Steps


#
# line segment intersection using vectors
# see Computer Graphics by F.S. Hill
#
def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return
def seg_intersect(a1, a2, b1, b2):
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    if denom.astype(float) == 0:
        return None
    return (num / denom.astype(float)) * db + b1


def normalize_times(data):
    return [[d[0] - data[0][0], d[1]] for d in data]


def compute_acceleration_magnitude(accelerometer_data):
    return normalize_times(
        [[int(timestamp) / NANOSECOND_TO_SECOND_FACTOR, norm((values["x"], values["y"], values["z"]))] for
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


def compute_moving_average_filter(moving_median_filter, window_length):
    return [[point[0], object_mean(lambda pt: pt[1], moving_median_filter[max(0, i - (window_length - 1)):i + 1])]
            for i, point in enumerate(moving_median_filter)]


def compute_moving_average_filter_aL(xs, ys, zs, window_length):
    return [mean(xs[max(0, i - (window_length - 1)):i + 1]) for i, point in enumerate(xs)], \
           [mean(ys[max(0, i - (window_length - 1)):i + 1]) for i, point in enumerate(ys)], \
           [mean(zs[max(0, i - (window_length - 1)):i + 1]) for i, point in enumerate(zs)]


def compute_threshold(data):
    return [[point[0], max(1.033, data[i - 4 if i - 4 > 0 else i][1])] for i, point in enumerate(data)]




def count_steps(data, lambdaD, axL, ayL, azL):
    steps = 0
    num_peaks = 0
    armed = False
    pos_cross_index = None
    lastValidStepIndex = None
    lastValidStepTime = None

    pos_cross_pt = None
    neg_cross_pt = None

    checked = False

    peaks = []
    crossNeg = []
    crossPos = []

    crossNegPt = []
    crossPosPt = []

    for i, point in enumerate(data):
        if point[1] > lambdaD[i][1] \
                and not armed \
                and (axL[i] < A_xmin or ayL[i] > A_ymin or azL[i] < A_zmin or azL[i] > A_zmin):

            armed = True
            num_peaks += 1

            crossPosPt.append(point)
            pos_cross_pt = seg_intersect(np.array(point), np.array(data[i - 1]), np.array(lambdaD[i]),
                                         np.array(lambdaD[i - 1]))
            crossPos.append(pos_cross_pt)

            pos_cross_index = i

        elif point[1] < lambdaD[i][1] and armed:

            neg_cross_pt = seg_intersect(np.array(point), np.array(data[i - 1]), np.array(lambdaD[i]),
                                         np.array(lambdaD[i - 1]))

            crossNeg.append(neg_cross_pt)
            crossNegPt.append(point)

            neg_cross_index = i
            step_interval = data[pos_cross_index:i + 1]

            peak = max(step_interval, key=lambda pt: pt[1])

            if num_peaks >= St_min:  # Rule 4
                # Rule 5, 7+8, 9
                if (len(peaks) == 0 or (len(peaks) > 0 and T_stmin < peak[0] - peaks[len(peaks)-1][0] < T_stmax)) \
                        and Acc_min <= max(data[pos_cross_index:neg_cross_index], key=lambda pt: pt[1])[1] <= Acc_max \
                        and neg_cross_pt[0] - pos_cross_pt[0] > T:

                    peaks.append(peak)

                    lastValidStepIndex = data.index(peak)

                    steps += 1

                else:
                    num_peaks = 0

            armed = False

        elif not armed:
            # Rules 2
            if lastValidStepIndex is not None and data[i][0] - data[lastValidStepIndex][0] > T_stmax and num_peaks != 0:
                num_peaks = 0
                # Rules 3.2, 3.3, 3.4
            if num_peaks >= St_min and steps > 0 and data[i][0] - data[lastValidStepIndex][0] >= T_stmax:
                num_peaks = 0
                steps -= 1

    return steps, peaks, crossPos, crossNeg, crossPosPt, crossNegPt


if __name__ == '__main__':
    with open('pach-cardiac-Accelerometer-export-20-steps.json') as f:
        accelerometer_data = json.load(f)

    a_x = [values["x"] for _, values in accelerometer_data.items()]
    a_y = [values["y"] for _, values in accelerometer_data.items()]
    a_z = [values["z"] for _, values in accelerometer_data.items()]

    a_xL, a_yL, a_zL = compute_moving_average_filter_aL(a_x, a_y, a_z, 16)

    # pprint(a_xL)
    # pprint(a_yL)
    # pprint(a_zL)

    # plt.plot(a_xL)
    # plt.plot(a_yL)
    # plt.plot(a_zL)

    # pprint(a_xL, a_yL, a_zL)

    Am = compute_acceleration_magnitude(accelerometer_data)
    AmL = compute_moving_median_filter(Am)
    # aL = compute_moving_average_filter_aL(accelerometer_data, 16)

    moving_avg_filter = compute_moving_average_filter(AmL, 8)

    AmH = [[pt[0], pt[1] - moving_avg_filter[i][1]] for i, pt in enumerate(AmL)]

    threshold = compute_threshold(AmL)
    thresholdavg = compute_threshold(moving_avg_filter)

    plt.plot(*zip(*AmL))
    # plt.plot(*zip(*AmL), '-o')

    # plt.plot(*zip(*Am))
    # plt.plot(*zip(*moving_avg_filter))
    plt.plot(*zip(*threshold), "-.")
    # plt.plot(*zip(*AmH))

    # pprint(Am)
    # pprint(AmL)

    num_step, peaks, crossPos, crossNeg, crossPosPt, crossNegPt = count_steps(AmL, threshold, a_xL, a_yL, a_zL)

    plt.plot(*zip(*peaks), "o")
    plt.plot(*zip(*crossPos), "o")
    plt.plot(*zip(*crossNeg), "o")
    plt.plot(*zip(*crossNegPt), "o")
    plt.plot(*zip(*crossPosPt), "o")

    plt.show()

    print("Number of Steps from AmL: {}".format(num_step))
    # print("Number of Steps from Mean Filter: {}".format(count_steps(moving_avg_filter, threshold)))
    # print("Number of Steps from Mean Filter: {}".format(count_steps(AmH, threshold)))
