import GPUtil
import threading
import time
import numpy as np


def hwhm_to_sigma(hwhm):
    return hwhm / (np.sqrt(2 * np.log(2)))


def sigma_to_hwhm(sigma):
    return sigma * (np.sqrt(2 * np.log(2)))


class Monitor(threading.Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay  # Time between calls to GPUtil
        self.start()

    def run(self):
        while not self.stopped:
            GPUtil.showUtilization()
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True