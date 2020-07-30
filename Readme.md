# Mjolnir (Thor's Camera)

A suite of (fairly minimal) programs to replace ThorCam, built using pyqtgraph.

The `mjolnir` GUI can connect to either of the following cameras:
* C1285R12M (which Thorlabs sells as [DCC1545M](https://www.thorlabs.com/thorproduct.cfm?partnumber=DCC1545M))
* [CS165MU1](https://www.thorlabs.com/thorproduct.cfm?partnumber=CS165MU1)

You can connect directly via USB, or you can use the server script to send images to GUI subscribers over a local network.

This version of Mjolnir uses [sipyco](https://github.com/m-labs/sipyco) for its remote procedure calls (RPCs).


## Installation

Typically installation will use [conda](https://anaconda.org/) to provide a segregated python environment.
This version is compatible with Python 3.7.

First, use pip to install [sipyco](https://github.com/m-labs/sipyco):

`pip install git+https://github.com/m-labs/sipyco`

Then use pip to install mjolnir into your environment:

`pip install git+https://github.com/OxfordIonTrapGroup/mjolnir`

The software currently uses the Thorlabs DLLs (on Windows) that are installed when installing Thorcam.

For Thorlabs DCx cameras, make sure the DLL (usually located at `C:\Program Files\Thorlabs\Scientific Imaging\DCx Camera Support\USB Driver Package\uc480_64.dll` for Windows 64-bit) is on your system path, otherwise ctypes won't be able to find it.

For Thorlabs Scienfitic Imaging cameras, it is recommended that the user install the latest revision of the Windows SDK from [Thorlabs](https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam).
After installation, move the folder to the appropriate path (usually `C:\Program Files\Thorlabs\Scientific Imaging\Scientific Camera Support` for Windows).
Then follow the instructions in the Python README (usually located in `Scientific_Camera_Interfaces-Rev_G\Scientific Camera Interfaces`) to move the DLLs to the appropriate folder and install the Python SDK to your conda environment.

**For Thorlabs Scientific Imaging cameras, the user must specify the DLL folder path within `mjolnir\drivers\tsi\__init__.py`.**

Linux support is tenuous at best but uses libueye from [IDS](https://en.ids-imaging.com/download-ueye-lin64.html).


## Usage

Help is available by using `--help` with any of the commands.


### `mjolnir_launcher`

A small launcher window that you can use to enter the main GUI.
Its main use is when the camera is connected locally, as it can list the serial numbers of all USB connected cameras.

When invoking from the command line, there are no arguments:

`$ mjolnir_launcher`


### `mjolnir_gui`

The meat of the program.
Fitting of 2D Gaussians showing the position on the camera and the parameters of the fit.
A zoomed image of the beam, with a corresponding image of the fit residuals is also shown.
You can mark positions (but not with the mouse).
Then you will be given the displacement of the beam from the mark.

When invoking from the command line, you must supply the connection type (local or remote) and parameters related to that type, i.e. camera serial number for local or network arguments for remote. E.g.:

`$ mjolnir_gui remote --server 127.0.0.1 --artiq-port 4000 --zmq-port 5555`


### `mjolnir_server`

Makes a single camera available on the network for GUIs to connect to.
The default ports are listed here and can be omitted.

`$ mjolnir_server --device <serial> --server 127.0.0.1 --artiq-port 4000 --zmq-port 5555`


### Tools

The functions used for fitting in the GUI can be used standalone on any monochrome image (provided it's suitably arranged as a numpy array).
The fit function is slow when used on large images due to the number of function evaluations - methods are provided for sensibly cropping and downsampling images such that the fit is much faster.


## Development

### Building executables

Building packaged executables is generally a pain in the neck - I wouldn't bother!

My own thoughts on this: conda itself is quite poor at keeping packages segregated when building executables with PyInstaller (I ended up resorting to virtualenv instead of conda).
Python 3.6 and PyInstaller 3.4 seem to play nicely together.
The built executable should be around 80MB.

Install ARTIQ, strip out all the packages we don't need, then install `mjolnir`.

The only lines in `mjolnir` containing ARTIQ imports are:

* `from artiq.protocols.pc_rpc import simple_server_loop`
* `from artiq.tools import verbosity_args, simple_network_args, init_logger`
* `from artiq.protocols.pc_rpc import Client`

`artiq.protocols` is well isolated, but `artiq.tools` imports things from elsewhere in the codebase that spiral into lots of things being imported unnecessarily.
Comment out the `is_experiment` import and any function you find it in!


### New features?

If `mjolnir` looks useful to you, feedback is appreciated!
Please use the GitHub issue tracker to raise bugs/improvements.


## Acknowledgements

The driver for the cameras (uc480) was taken from: <https://github.com/ddietze/pyUVVIS>.
This appears to no longer be maintained.
The driver uses the Thorlabs DLL mentioned previously.
