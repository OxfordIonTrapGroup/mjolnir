#!/usr/bin/env python3
#
# Minimal test script for mjolnir's GUI
# Uses dummy camera as substitute for locally connected camera
#
import sys
from PyQt5 import QtWidgets

from mjolnir.ui.beam import BeamDisplay
from mjolnir.test.dummy_zmq import DummyCamera


def main():
    app = QtWidgets.QApplication(sys.argv)

    camera = DummyCamera()
    b = BeamDisplay(camera)
    camera.register_callback(lambda im: b.queue_image(im))

    sys.exit(app.exec_())
