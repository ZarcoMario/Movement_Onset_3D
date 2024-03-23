'''
Fligge, N., McIntyre, J., & van der Smagt, P. (2012, June).
Minimum jerk for human catching movements in 3D.
In 2012 4th IEEE RAS & EMBS International Conference on Biomedical Robotics and Biomechatronics (BioRob)
(pp. 581-586). IEEE.
https://mediatum.ub.tum.de/doc/1285786/document.pdf
'''
import numpy as np


def minimum_jerk_3d(x_ic: np.array, y_ic: np.array, z_ic: np.array,
                    x_fc: np.array, y_fc: np.array, z_fc: np.array,
                    d: float, t: np.array) -> np.array:
    x = _a0(x_ic) + _a1(x_ic) * t + _a2(x_ic) * t**2 + _a3(x_ic, x_fc, d) * t**3 \
        + _a4(x_ic, x_fc, d) * t**4 + _a5(x_ic, x_fc, d) * t**5
    y = _a0(y_ic) + _a1(y_ic) * t + _a2(y_ic) * t**2 + _a3(y_ic, y_fc, d) * t**3 \
        + _a4(y_ic, y_fc, d) * t**4 + _a5(y_ic, y_fc, d) * t**5
    z = _a0(z_ic) + _a1(z_ic) * t + _a2(z_ic) * t**2 + _a3(z_ic, z_fc, d) * t**3 \
        + _a4(z_ic, z_fc, d) * t**4 + _a5(z_ic, z_fc, d) * t**5
    return x, y, z


def _a0(x_ic):
    return x_ic[0]


def _a1(x_ic):
    return x_ic[1]


def _a2(x_ic):
    return 0.5 * x_ic[2]


def _a3(x_ic, x_fc, d):
    return (-10/d**3) * x_ic[0] + (-6/d**2) * x_ic[1] + (-3/(2*d)) * x_ic[2] \
           + (10/d**3) * x_fc[0] + (-4/d**2) * x_fc[1] + (1/(2*d)) * x_fc[2]


def _a4(x_ic, x_fc, d):
    return (15/d**4) * x_ic[0] + (8/d**3) * x_ic[1] + (3/(2*d**2)) * x_ic[2] \
           + (-15/d**4) * x_fc[0] + (7/d**3) * x_fc[1] + (-1/d**2) * x_fc[2]


def _a5(x_ic, x_fc, d):
    return (-6/d**5) * x_ic[0] + (-3/d**4) * x_ic[1] + (-1/(2*d**3)) * x_ic[2] \
           + (6/d**5) * x_fc[0] + (-3/d**4) * x_fc[1] + (1/(2*d**3)) * x_fc[2]