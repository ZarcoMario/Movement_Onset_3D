'''
Change the return values of the functions according to the requirements of the GUI
TODO: Update the docstring with the new arguments
'''

import numpy as np
from scipy.optimize import leastsq
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec



def onset_detection(m: int, position_x: np.array, position_y: np.array, position_z: np.array,
                    time: np.array, velocity_x: np.array, velocity_y: np.array, velocity_z: np.array,
                    t_th: float=np.inf, vel_th: float=0.80, debug: bool=False) -> tuple[float, dict, bool, bool]:
    """
    Two-dimensional movement onset time detection method. The function finds the maximum velocity
        before calling the function that finds initiation time.
        The search for the two consecutive equal-sized segments that fit the model is performed before
        the time corresponding to a percentage (vel_th) of the maximum velocity.
        This function has a temporal condition (t_th): if the minimum was found after the time in which
        the controller left the starting sphere, the previous minimum is selected if exists.
    :param m: length of the segments (m samples per each segment)
    :param position_x: trajectory data along a first dimension (e.g. x)
    :param position_y: trajectory data along a second dimension (e.g. z)
    :param time: time corresponding to trajectory data
    :param velocity_x: first derivative of position_x
    :param velocity_y: first derivative of position_y
    :param t_th: temporal threshold (e.g. time corresponding to the moment in which the controller left the starting sphere)
    :param vel_th: percentage of max velocity (vel_th < 1)
    :return:
        - t_onset: Initiation time
        - dict_results: Different values that can be used for validation and analysis
        - converged: True is the default value. False if minimum was not found
        - adjusted_t: False is the default value. True if the minimum was adjusted based on the time-threshold condition.
    """

    # Find Max Velocity along all dimension to only consider the trajectory before the velocity threshold
    # This step is not necessary if another type of threshold is to be used

    speed_x, speed_y, speed_z = np.abs(velocity_x), np.abs(velocity_y), np.abs(velocity_z)

    max_vx = np.max(speed_x)
    max_vy = np.max(speed_y)
    max_vz = np.max(speed_z)

    max_vel = np.max([max_vx, max_vy, max_vz])

    if max_vel == max_vx:
        max_v, speed = max_vx, speed_x
    elif max_vel == max_vy:
        max_v, speed = max_vy, speed_y
    else:
        max_v, speed = max_vz, speed_z

    idx = np.argwhere(speed == max_v)[0][0]
    max_v = speed[idx]
    indexes = np.argwhere(speed <= vel_th * max_v).T[0]

    if debug:

        fig = plt.figure()
        gs = GridSpec(1, 2)

        ax = fig.add_subplot(gs[0, 0])
        ax.grid(True)
        ax.plot(time, velocity_x, '.-', label='vx')
        ax.plot(time, velocity_y, '.-', label='vy')
        ax.plot(time, velocity_z, '.-', label='vz')
        ax.legend()
        ax.set_xlabel("t")

        ax = fig.add_subplot(gs[0, 1])
        ax.grid(True)
        ax.plot(time, speed_x, '.-', label='|vx|')
        ax.plot(time, speed_y, '.-', label='|vy|')
        ax.plot(time, speed_z, '.-', label='|vz|')
        ax.plot(time[idx], speed[idx], 'ro')
        ax.legend()
        ax.set_xlabel("t")

        # plt.show()

    # In case indexes are not consecutive
    diffs = np.diff(indexes) != 1
    idx = np.nonzero(diffs)[0] + 1
    groups = np.split(indexes, idx)
    indexes = groups[0]

    position_x = position_x[indexes]
    position_y = position_y[indexes]
    position_z = position_z[indexes]
    time = time[indexes]

    t_onset, dict_results, converged, adjusted_t = _movement_onset(m, position_x, position_y, position_z, time, t_th, debug)
    dict_results['max_vel'] = max_v
    dict_results['indexes'] = indexes

    return t_onset, dict_results, converged, adjusted_t


def _movement_onset(m: int, position_x: np.array, position_y: np.array, position_z: np.array, time: np.array,
                    t_th: float, debug: bool):
    """
    A two-dimensional  extension of the method found here
        https://www.frontiersin.org/articles/10.3389/neuro.20.002.2009/full
    :param m: length of one segment (m samples)
    :param position_x: trajectory data along a first dimension (e.g. x)
    :param position_y: trajectory data along a second dimension (e.g. z)
    :param time: time corresponding to trajectory data
    :param velocity_x: first derivative of position_x
    :param velocity_y: first derivative of position_y
    :param t_th: temporal threshold (e.g. time corresponding to the moment in which the controller left the starting sphere)
    :return:
        - t_onset: Initiation time
        - dict_results: Different values that can be used for validation and analysis
        - converged: True is the default value. False if minimum was not found
        - adjusted_t: False is the default value. True if the minimum was adjusted based on the time-threshold condition.
    """
    # Check time series are the same length
    # Time series should never be of different length but just in case
    try:
        assert position_x.size == time.size, "Time series are of different size"
    except AssertionError as msg:
        print(msg)

    try:
        assert position_y.size == time.size, "Time series are of different size"
    except AssertionError as msg:
        print(msg)

    try:
        assert position_z.size == time.size, "Time series are of different size"
    except AssertionError as msg:
        print(msg)

    adjusted_t = False

    jerks = []
    errors = np.array([])
    times = np.array([])
    onsets = np.array([])
    jerks_mean = []

    x_s1 = []
    y_s1 = []
    z_s1 = []
    t_s1 = []

    x_s2 = []
    y_s2 = []
    z_s2 = []
    t_s2 = []

    for i in range(time.size - 2*m + 1):

        # First Segment
        xl_1 = position_x[i:m+i]
        yl_1 = position_y[i:m+i]
        zl_1 = position_z[i:m+i]
        tl_1 = time[i:m+i]

        x_s1.append(xl_1)
        y_s1.append(yl_1)
        z_s1.append(zl_1)
        t_s1.append(tl_1)

        # Second Segment
        xl_2 = position_x[m+i-1:2*m+i-1]
        yl_2 = position_y[m+i-1:2*m+i-1]
        zl_2 = position_z[m+i-1:2*m+i-1]
        tl_2 = time[m+i-1:2*m+i-1]

        x_s2.append(xl_2)
        y_s2.append(yl_2)
        z_s2.append(zl_2)
        t_s2.append(tl_2)

        hat_t_q = tl_1[-1]
        times = np.append(times, hat_t_q)

        # ----------- #
        # ---- X ---- #
        # ----------- #

        # Static phase
        hat_x_q = np.mean(xl_1)

        # Movement phase
        def minimum_jerk(Um, x2, t2):
            f = x2 - (hat_x_q + Um * (t2 - hat_t_q) ** 3)
            return (1 / m ** 0.5) * f

        U_i = np.asarray([0])  # Initial value for optimization
        result = leastsq(minimum_jerk, U_i, (xl_2, tl_2))
        Um_x = result[0][0]

        # ----------- #
        # ---- Y ---- #
        # ----------- #

        # Static phase
        hat_y_q = np.mean(yl_1)

        # Movement phase
        def minimum_jerk(Um, y2, t2):
            f = y2 - (hat_y_q + Um * (t2 - hat_t_q) ** 3)
            return (1 / m ** 0.5) * f

        U_i = np.asarray([0])  # Initial value for optimization
        result = leastsq(minimum_jerk, U_i, (yl_2, tl_2))
        Um_y = result[0][0]

        # ----------- #
        # ---- Z ---- #
        # ----------- #

        # Static phase
        hat_z_q = np.mean(zl_1)

        # Movement phase
        def minimum_jerk(Um, z2, t2):
            f = z2 - (hat_z_q + Um * (t2 - hat_t_q) ** 3)
            return (1 / m ** 0.5) * f

        U_i = np.asarray([0])  # Initial value for optimization
        result = leastsq(minimum_jerk, U_i, (zl_2, tl_2))
        Um_z = result[0][0]

        jerks.append(np.array([Um_x, Um_y, Um_z]))

        # --------------- #
        # ---- ERROR ---- #
        # --------------- #

        error_x = np.power(xl_1 - hat_x_q, 2).sum() + np.power(xl_2 - hat_x_q - Um_x * (tl_2 - hat_t_q) ** 3, 2).sum()
        error_y = np.power(yl_1 - hat_y_q, 2).sum() + np.power(yl_2 - hat_y_q - Um_y * (tl_2 - hat_t_q) ** 3, 2).sum()
        error_z = np.power(zl_1 - hat_z_q, 2).sum() + np.power(zl_2 - hat_z_q - Um_z * (tl_2 - hat_t_q) ** 3, 2).sum()

        error = (error_x + error_y + error_z) ** 0.5
        errors = np.append(errors, error)

        # --------------- #
        # ---- ONSET ---- #
        # --------------- #

        def rms_error(t_onset, x1, x2, y1, y2, z1, z2, t2):
            f_x = np.power(x1 - hat_x_q, 2).sum() + np.power(x2 - hat_x_q - Um_x * (t2 - t_onset) ** 3, 2).sum()
            f_y = np.power(y1 - hat_y_q, 2).sum() + np.power(y2 - hat_y_q - Um_y * (t2 - t_onset) ** 3, 2).sum()
            f_z = np.power(z1 - hat_z_q, 2).sum() + np.power(z2 - hat_z_q - Um_z * (t2 - t_onset) ** 3, 2).sum()
            f = f_x + f_y + f_z
            return f**0.5

        to = np.asarray([0])  # Initial value for optimization
        result = leastsq(rms_error, to, (xl_1, xl_2, yl_1, yl_2, zl_1, zl_2, tl_2))
        t_mo = result[0][0]
        onsets = np.append(onsets, t_mo)

        # --------------- #
        # ---- JERKS ---- #
        # --------------- #

        def mean_jerk(Um, x2, t2):
            f = x2 - (hat_x_q + Um * (t2 - t_mo) ** 3)
            return (1 / m ** 0.5) * f

        U_i = np.asarray([0])  # Initial value for optimization
        result = leastsq(mean_jerk, U_i, (xl_2, tl_2))
        U_mean_x = result[0][0]

        def mean_jerk(Um, y2, t2):
            f = y2 - (hat_y_q + Um * (t2 - t_mo) ** 3)
            return (1 / m ** 0.5) * f

        U_i = np.asarray([0])  # Initial value for optimization
        result = leastsq(mean_jerk, U_i, (yl_2, tl_2))
        U_mean_y = result[0][0]

        def mean_jerk(Um, z2, t2):
            f = z2 - (hat_z_q + Um * (t2 - t_mo) ** 3)
            return (1 / m ** 0.5) * f

        U_i = np.asarray([0])  # Initial value for optimization
        result = leastsq(mean_jerk, U_i, (zl_2, tl_2))
        U_mean_z = result[0][0]

        jerks_mean.append(np.array([U_mean_x, U_mean_y, U_mean_z]))

    if errors.size != 0:

        peaks, _ = find_peaks(-errors)

        if peaks.size == 0:
            # No minimum was found
            converged = False
            min_error = np.min(errors[np.nonzero(errors)])
        else:
            # First minimum
            converged = True
            min_error = errors[peaks][-1]
        index = np.argwhere(errors == min_error)[0][0]
        t_onset = onsets[index]

        # Adjust t_onset if it is higher than t_th
        i = 1
        while t_onset > t_th and errors[peaks].size - i > 0:
            adjusted_t = True
            min_error = errors[peaks][-1 - i]
            index = np.argwhere(errors == min_error)[0][0]
            t_onset = onsets[index]
            i += 1

        if debug:

            fig = plt.figure()
            gs = GridSpec(1, 1)

            ax = fig.add_subplot(gs[0, 0])
            ax.grid(True)
            ax.plot(times, errors, '.-')
            ax.plot(times[index], errors[index], 'ro')
            ax.set_xlabel("t")
            ax.set_ylabel("error")

            plt.show()

        jerk_mean = jerks_mean
        x1 = x_s1[index]
        y1 = y_s1[index]
        z1 = z_s1[index]
        t1 = t_s1[index]
        x2 = x_s2[index]
        y2 = y_s2[index]
        z2 = z_s2[index]
        t2 = t_s2[index]

        dict_results = {
            'Um': jerk_mean,
            'min_error': min_error,
            'errors': errors,
            'times': times,
            'x1': x1,
            'y1': y1,
            'z1': z1,
            't1': t1,
            'x2': x2,
            'y2': y2,
            'z2': z2,
            't2': t2
        }

    else:

        t_onset = None
        dict_results = {}
        converged = False

    return t_onset, dict_results, converged, adjusted_t

