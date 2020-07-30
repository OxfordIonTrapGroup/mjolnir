#!/usr/bin/env python3.5
import argparse
import sys
import time
import zmq

from sipyco.pc_rpc import simple_server_loop
from sipyco import simple_network_args, init_logger
from mjolnir.drivers.camera import Camera, list_serial_numbers

# Copy pasted from oxart, since that isn't publicly available
# verbosity_args() was renamed to add_common_args() in ARTIQ 5.0; support both.
try:
    from artiq.tools import add_common_args
except ImportError:
    from artiq.tools import verbosity_args as add_common_args


def get_argparser():
    parser = argparse.ArgumentParser()
    simple_network_args(parser, 4000)
    add_common_args(parser)
    parser.add_argument("--list", action="store_true",
        help="list connected cameras (ignores all other arguments)")
    parser.add_argument("--broadcast-images", action="store_true")
    parser.add_argument("--zmq-bind", default="*")
    parser.add_argument("--zmq-port", default=5555, type=int)
    parser.add_argument("--device", "-d", type=int, default=None,
        help="camera serial number, uses first available if not supplied")
    return parser


def create_zmq_server(bind="*", port=5555):
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.set_hwm(1)
    socket.bind("tcp://{}:{}".format(bind, port))
    return socket


def run_server(dev, args):
    if args.broadcast_images:
        socket = create_zmq_server(args.zmq_bind, args.zmq_port)
        dev.register_callback(lambda im: socket.send_pyobj(im))

    sn = dev.get_serial_no()
    try:
        simple_server_loop({"thorcam sn:{}".format(sn): dev}, args.bind, args.port)
    finally:
        dev.close()


def main():
    args = get_argparser().parse_args()
    init_logger(args)

    if args.list:
        serials = list_serial_numbers()
        print(serials if serials else "No cameras found")
        return

    dev = Camera(sn=args.device)
    run_server(dev, args)


if __name__ == "__main__":
    main()
