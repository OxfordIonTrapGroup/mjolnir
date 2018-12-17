#!/usr/bin/env python3

from setuptools import setup, find_packages
import sys


if sys.version_info[:3] < (3, 5, 3):
    raise Exception("You need Python 3.5.3+")


requirements = [
    'numpy',
    'scipy',
    'zmq',
    'pyqt5',
    'pyqtgraph',
    'quamash'
]

console_scripts = [
    "mjolnir_server=mjolnir.frontend.server:main",
]

gui_scripts = [
    "mjolnir_gui=mjolnir.frontend.gui:main",
    "mjolnir_launcher=mjolnir.frontend.launcher:main",
]


setup(
    name='mjolnir',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts":console_scripts,
        "gui_scripts":gui_scripts,
    }
)
