#!/usr/bin/env python3.5
import zmq
import sys
import argparse
import atexit
import logging
from PyQt5 import QtWidgets, QtCore
from sipyco.pc_rpc import Client

from mjolnir.ui.beam import BeamDisplay
from mjolnir.drivers.camera import Camera, list_serial_numbers


logger = logging.getLogger(__name__)


def zmq_setup(ctx, server, port):
    sock = ctx.socket(zmq.SUB)
    sock.set_hwm(1)
    sock.connect("tcp://{}:{}".format(server, port))
    sock.setsockopt_string(zmq.SUBSCRIBE, '')
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt(zmq.RCVTIMEO, 1)
    return sock


def remote(args):
    ### Remote operation ###
    camera = Client(args.server, args.artiq_port)
    sn = camera.get_serial_no()
    b = BeamDisplay(camera)
    ctx = zmq.Context()
    sock = zmq_setup(ctx, args.server, args.zmq_port)

    @QtCore.pyqtSlot()
    def qt_update():
        try:
            im = sock.recv_pyobj()
        except zmq.error.Again as e:
            pass
        else:
            b.queue_image(im)

    timer = QtCore.QTimer(b)
    timer.timeout.connect(qt_update)
    timer.start(50) # timeout ms

    title = b.windowTitle() + " (sn: {} @ host: {}, artiq: {}, zmq: {})".format(
        sn, args.server, args.artiq_port, args.zmq_port)
    b.setWindowTitle(title)


def local(args):
    ### Local operation ###
    if args.list:
        serials = list_serial_numbers()
        print(serials if serials else "No cameras found")
        sys.exit()

    camera = Camera(args.device)
    sn = camera.get_serial_no()
    b = BeamDisplay(camera)
    camera.register_callback(lambda im: b.queue_image(im))
    atexit.register(camera.close)

    title = b.windowTitle() + " (sn: {} @ local)".format(sn)
    b.setWindowTitle(title)


def get_argparser():
    parser = argparse.ArgumentParser(description="GUI for Thorlabs CMOS cameras")
    subparsers = parser.add_subparsers()

    remote_parser = subparsers.add_parser("remote",
        help="connect to a camera providing a ZMQ/Artiq network interface")
    remote_parser.add_argument("--server", "-s", type=str, required=True)
    remote_parser.add_argument("--artiq-port", "-p", type=int, default=4000)
    remote_parser.add_argument("--zmq-port", "-z", type=int, default=5555)
    remote_parser.set_defaults(func=remote)

    local_parser = subparsers.add_parser("local",
        help="connect directly to a local camera")
    local_parser.add_argument("--list", action="store_true",
        help="list connected cameras (ignores all other arguments)")
    local_parser.add_argument("--device", "-d", type=int, default=None,
        help="camera serial number, uses first available if not supplied")
    local_parser.set_defaults(func=local)

    return parser


def main():
    parser = get_argparser()

    args = parser.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    args.func(args)

    logger.debug("Starting UI")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
