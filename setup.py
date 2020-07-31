#!/usr/bin/env python3

from setuptools import setup, find_packages
import sys


if (sys.version_info[:3] < (3, 5, 3)) or (sys.version_info[:3] > (3, 7, 7)):
    raise Exception("You need Python 3.5.3 - 3.7.7")


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
    "mjolnir_gui=mjolnir.frontend.gui:main",
    "mjolnir_launcher=mjolnir.frontend.launcher:main",
]


setup(
    name='mjolnir',
    version='0.2',
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts":console_scripts,
    }
)
