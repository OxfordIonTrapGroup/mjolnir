pyinstaller profiler_gui.py -y --paths $CONDA_PREFIX/Lib/site-packages/PyQt5/Qt/bin -d --exclude-module matplotlib --paths $CONDA_PREFIX/Lib/site-packages --paths $CONDA_PREFIX/Lib/site-packages/Library/bin --onefile
