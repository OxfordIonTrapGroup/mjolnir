# Additional Notes

There are some dependencies not currently included:

* matplotlib
* mpl_toolits

These are used in the test scripts for 3D plotting, but this
is not necessary for actual operation


## Saving images

This requires `pypng`.
I haven't implemented this yet.


## DLL Location

Failed to install python properly so I had some issues getting the DLL loaded.
This is the install location so I don't have to look for it again.

`C:\Program Files\Thorlabs\Scientific Imaging\uc480_64.dll`


## Building with PyInstaller

Didn't look hard enough at what was supported when trying to build.

~~
PyInstaller 3.4 (9 Sep 2018 release) isn't compatible with Python 3.7 at this
point in time, so pip installed from github, develop branch, at this commit:
https://github.com/pyinstaller/pyinstaller/commit/5edb4f7dee9c21f77c4a89342eb98248fe4c45b9
~~

Use Python 3.6 and PyInstaller 3.4, and it should 'just work'.


## Improvements

Things I know should be better but aren't.

* Would be really nice to have the 1D slices overlaid on the image itself but
  I haven't figured out how to do this yet. When the plot is added directly,
  it only appears at the edge of the image.
* Implement ROI selection
* add icon! The following works for apps launched from Python, but not
  PyInstaller:

```python
icon = QtGui.QIcon()
icon.addFile(pkg_resources.resource_filename(
    "mjolnir.resources", "icon.svg"))
self.setWindowIcon(icon)
```

* Fixed. ~~**BUG** The image fitting works great, but the lines appear half a
  pixel above and to the left of where they should. This is because the pixel
  is drawn to the bottom right of its index~~


### Camera interfacing

At the moment uses a random person's interface to the camera DLL. Thorlabs
cameras are rebranded IDS uEye cameras - there is a pyuEye package available
from IDS which may prove useful, as interfacing with C is a little annoying.
