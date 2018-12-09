import numpy as np
from PyQt5 import QtCore


def update_rate(last, ups=None):
    """Recalculate update rate given last update time and rate

    Intended usage is e.g. frames per second counter
    last, fps = update_rate(last, fps)

    :param last: QtCore.QTime object giving last update time
    :param ups: last update rate, or None if unknown
    :returns: now, update_rate
    """
    now = QtCore.QTime.currentTime()
    dt = float(last.msecsTo(now))/1000
    if ups is None:
        ups = 1.0 / dt
    else:
        s = np.clip(dt*3., 0, 1)
        ups = ups * (1-s) + (1.0/dt) * s

    return now, ups
