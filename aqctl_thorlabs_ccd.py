#!/usr/bin/env python3.5
import argparse
import sys
import time
import zmq

from artiq.protocols.pc_rpc import simple_server_loop
from artiq.tools import verbosity_args, simple_network_args, init_logger
from camera_controller.camera_controller import ThorlabsCCD

def get_argparser():
    parser = argparse.ArgumentParser()
    simple_network_args(parser, 4000)
    verbosity_args(parser)
    parser.add_argument("--broadcast-images", action="store_true")
    parser.add_argument("--zmq-bind", default="*")
    parser.add_argument("--zmq-port", default=5555, type=int)
    return parser

def create_zmq_server(bind="*", port=5555):
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.set_hwm(1)
    socket.bind("tcp://{}:{}".format(bind, port))
    return socket

def main():
    args = get_argparser().parse_args()
    init_logger(args)

    dev = ThorlabsCCD()

    if args.broadcast_images:
        socket = create_zmq_server(args.zmq_bind, args.zmq_port)
        dev.register_callback(lambda im: socket.send_pyobj(im))

    try:
        simple_server_loop({"camera": dev}, args.bind, args.port)
    finally:
        dev.close()

if __name__ == "__main__":
    main()
