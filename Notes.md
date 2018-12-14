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


### Building with PyInstaller

Didn't look hard enough at what was supported when trying to build.

~~
PyInstaller 3.4 (9 Sep 2018 release) isn't compatible with Python 3.7 at this
point in time, so pip installed from github, develop branch, at this commit:
https://github.com/pyinstaller/pyinstaller/commit/5edb4f7dee9c21f77c4a89342eb98248fe4c45b9
~~

Use Python 3.6 and PyInstaller 3.4, and it should 'just work'.
