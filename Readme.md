# Mjolnir (Thor's Camera)

A suite of (fairly minimal) programs to replace ThorCam, built using pyqtgraph.

The `mjolnir` GUI can connect to either of the following cameras:
* C1285R12M (which Thorlabs sells as [DCC1545M](https://www.thorlabs.com/thorproduct.cfm?partnumber=DCC1545M))
* [CS165MU1](https://www.thorlabs.com/thorproduct.cfm?partnumber=CS165MU1)

You can connect directly via USB, or you can use the server script to send images to GUI subscribers over a local network.

This version of Mjolnir uses [sipyco](https://github.com/m-labs/sipyco) for its remote procedure calls (RPCs).


## Installation

Typically installation will use [conda](https://anaconda.org/) to provide a segregated python environment.
This version is compatible with Python 3.5.3 - 3.7.7.

First, install git to your environment:

`conda install git`

Next, use pip to install [sipyco](https://github.com/m-labs/sipyco):

`python -m pip install git+https://github.com/m-labs/sipyco`

Then use pip to install mjolnir into your environment:

`python -m pip install git+https://github.com/OregonIons/mjolnir`

Next, it is required that the user completes the steps in **both** of the following two sub-sections, regardless of which camera is being used.


### Thorlabs DCx Cameras

The software currently uses the Thorlabs DLLs (on Windows) that are installed when installing Thorcam.

Make sure the uc480 DLL (usually located at `C:\Program Files\Thorlabs\Scientific Imaging\DCx Camera Support\USB Driver Package\uc480_64.dll` for Windows 64-bit) is on your system path, otherwise ctypes won't be able to find it.

Linux support is tenuous at best but uses libueye from [IDS](https://en.ids-imaging.com/download-ueye-lin64.html).


### Thorlabs Scientific Imaging Cameras

For Thorlabs Scientific Imaging cameras, it is recommended that the user install Revision G of the Scientific Camera Interface from [Thorlabs](https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam). The file is located under `Programming Interfaces -> Windows SDK and Doc. for Scientific Cameras`.

After installation, move the unzipped folder (called `Scientific_Camera_Interfaces-Rev_G`) to the appropriate path (`C:\Program Files\Thorlabs\Scientific Imaging\Scientific Camera Support` for Windows).

Then use pip to install the SDK to your environment:

`python -m pip install "C:\Program Files\Thorlabs\Scientific Imaging\Scientific Camera Support\Scientific_Camera_Interfaces-Rev_G\Scientific Camera Interfaces\SDK\Python Compact Scientific Camera Toolkit\thorlabs_tsi_camera_python_sdk_package.zip"`

**Finally, the user must specify the Scientific Camera DLL folder path within `mjolnir\drivers\tsi\__init__.py`.**

The folder path is usually `C:\Program Files\Thorlabs\Scientific Imaging\Scientific Camera Support\Scientific_Camera_Interfaces-Rev_G\Scientific Camera Interfaces\SDK\Native Compact Camera Toolkit\dlls`. The software automatically selects the Native_32_lib or Native_64_lib folder, depending on your system.

**Note:** Do not specify the uc480 DLL path. As stated in the previous sub-section, c-types will find it.


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


## Jupyter Notebook

The software has the ability to save and load individual frames.
It accomplishes this task by using the Python's pickle module to save/load a `.pickle` file containing the dictionary object that stores all relevant image and beam data.

A [Jupyter Notebook](https://github.com/OregonIons/mjolnir-frame-analyzer) was created to analyze saved frames.

This Notebook simply stores the data and re-creates the images from the GUI. It's purpose is to be used as a starting point for any further analysis that the user may want to perform.


## Development

### New features?

If `mjolnir` looks useful to you, feedback is appreciated!
Please use the GitHub issue tracker to raise bugs/improvements.


## Acknowledgements

The driver for the Thorlabs DCx cameras (uc480) was taken from: <https://github.com/ddietze/pyUVVIS>.
This appears to no longer be maintained.
The driver uses the Thorlabs DLL mentioned previously.
