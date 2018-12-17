pyinstaller mjolnir/test/gui_test.py -y -d all --clean --onefile --windowed -n mjolnir_test.exe &
pyinstaller mjolnir/frontend/launcher.py -y --clean --onefile --windowed -n mjolnir_launcher.exe &
pyinstaller mjolnir/frontend/server.py -y --clean --onefile -n mjolnir_server.exe &
pyinstaller mjolnir/frontend/gui.py -y --clean --onefile -n mjolnir_gui.exe &

# These aren't needed any more
# --exclude-module matplotlib
# --paths $VIRTUAL_ENV/Lib/site-packages/PyInstaller
# --paths $VIRTUAL_ENV/Lib/site-packages/PyQt5/Qt/bin  \
# --paths $VIRTUAL_ENV/Lib/site-packages/Library/bin   \
# --paths $VIRTUAL_ENV/Lib/site-packages

# You can try this but it didn't work for me
# (built the exe but it wouldn't run)
# --upx-dir=$HOME/scratch/upx-3.95-win64
