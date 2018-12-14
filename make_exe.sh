pyinstaller mjolnir/test/gui_test.py -y -d all --clean --onefile --windowed
#pyinstaller mjolnir/frontend/launcher.py -y --clean --onefile --windowed

# These aren't needed any more
# --exclude-module matplotlib
# --paths $VIRTUAL_ENV/Lib/site-packages/PyInstaller
# --paths $VIRTUAL_ENV/Lib/site-packages/PyQt5/Qt/bin  \
# --paths $VIRTUAL_ENV/Lib/site-packages/Library/bin   \
# --paths $VIRTUAL_ENV/Lib/site-packages
