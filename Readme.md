# Mjolnir (Thor's Camera)

A suite of (fairly minimal) programs to replace ThorCam, built
using pyqtgraph.
The GUI can connect to cameras directly via USB, or you can
use the server script to send images to GUI subscribers over
a local network.

Mjolnir uses ARTIQ for its remote procedure calls (RPCs).
Installing ARTIQ is overkill, since we only need the protocols,
but it's ubiquitous in the Oxford Ion Trap Group.
Sorry!
We're aiming to provide a minimal ARTIQ install with just the
RPC protocol soon which will make things much better.


## Installation

Follow the instructions for installing ARTIQ, then clone the
repo and pip install mjolnir into your environment.

Or if you just want the viewer (and not the server), download
the executable, once I manage to make it!


## Usage

Help is available by using `--help` with any of the commands.


### `mjolnir_launcher`

This is just a small launcher window that you can use to enter the
main GUI.
If you downloaded the executable, this is what will appear.
Its main use is when the camera is connected locally, as it will
(eventually once I implement this) list the serial numbers of all
USB connected cameras.

When invoking from the command line, there are no arguments:

`$ mjolnir_launcher`


### `mjolnir_gui`

The meat of the program.
Fitting of 2D Gaussians showing the position on the camera and the
parameters of the fit.
A zoomed image of the beam, with a corresponding image of the fit
residuals is also shown.
You can mark positions (but not with the mouse).
Then you will be given the displacement of the beam from the mark.

When invoking from the command line, you must supply the connection
type (local or remote) and parameters related to that type, i.e.
camera serial number for local or network arguments for remote. E.g.:

`$ mjolnir_gui remote --server 127.0.0.1 --artiq-port 4000 --zmq-port 5555`


### `mjolnir_server`

Makes a single camera available on the network for GUIs to connect to.
The default ports are listed here and can be omitted.

`$ mjolnir_server --device <serial> --server 127.0.0.1 --artiq-port 4000 --zmq-port 5555`


## Development

You probably want to use virtualenv as it seems to do a much
better job of isolating environments than conda.
Try to get a minimal ARTIQ installation, then pip install mjolnir.

The only lines containing ARTIQ imports are:

* `from artiq.protocols.pc_rpc import simple_server_loop`
* `from artiq.tools import verbosity_args, simple_network_args, init_logger`
* `from artiq.protocols.pc_rpc import Client`

`artiq.protocols` is well isolated, but `artiq.tools` imports things from
elsewhere in the codebase that spiral into lots of things being imported
unnecessarily.
Comment out the `is_experiment` import and any function you find it in!


### New features?

Ask.


## Acknowledgements

The driver for the cameras (uc480) was taken from:
https://github.com/ddietze/pyUVVIS .
This appears to no longer be maintained.
