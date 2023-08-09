# Mjolnir (Thor's Camera)

A suite of (fairly minimal) programs to replace ThorCam, built using pyqtgraph.
The Thorlabs cameras are just rebadged [IDS](https://en.ids-imaging.com/home.html) cameras, so IDS's software could be better, but I haven't tried it.
The `mjolnir` GUI can connect to cameras directly via USB, or you can use the server script to send images to GUI subscribers over a local network.

Mjolnir uses [sipyco](https://github.com/m-labs/sipyco) for its remote procedure calls (RPCs).


## Installation

The software currently uses the Thorlabs DLL (on Windows) that is installed when installing Thorcam.
Make sure the DLL (usually located at `C:\Program Files\Thorlabs\Scientific Imaging\uc480_64.dll` for Windows 64-bit) is on your system path, otherwise ctypes won't be able to find it.
Linux support is tenuous at best but uses libueye from [IDS](https://en.ids-imaging.com/download-ueye-lin64.html).


Typically installation will use [conda](https://anaconda.org/) to provide a segregated python environment.
Use pip to install `sipyco` and `mjolnir` into your environment:

`pip install git+https://github.com/m-labs/sipyco.git`
`pip install git+https://github.com/OxfordIonTrapGroup/mjolnir`


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


### New features?

If `mjolnir` looks useful to you, feedback is appreciated!
Please use the GitHub issue tracker to raise bugs/improvements.


## Acknowledgements

The driver for the cameras (uc480) was taken from: <https://github.com/ddietze/pyUVVIS>.
This appears to no longer be maintained.
