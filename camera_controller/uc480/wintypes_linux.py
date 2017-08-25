"""
.. module: uc480.wintypes_linux
   :platform: Windows, Linux
.. moduleauthor:: Daniel Dietze <daniel.dietze@berkeley.edu> 

Provides :py:mod:`ctypes.wintypes` identifiers for linux. The *wintypes* module seems not to be part of the :py:mod:`ctypes` library under Linux, so I have set up my own.

..
   This file is part of the uc480 python module.

   The uc480 python module is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   The uc480 python module is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with the uc480 python module. If not, see <http://www.gnu.org/licenses/>.

   Copyright 2015 Daniel Dietze <daniel.dietze@berkeley.edu>.   
""" 
import ctypes

BOOLEAN = ctypes.c_int32
BOOL = ctypes.c_int32
INT = ctypes.c_int32
UINT = ctypes.c_int32
LONG = ctypes.c_int32 
# VOID = void 
LPVOID = ctypes.c_void_p
ULONG = ctypes.c_uint32

UINT64 = ctypes.c_uint64
__int64 = ctypes.c_int64
LONGLONG = ctypes.c_int64
DWORD = ctypes.c_uint32 
WORD = ctypes.c_uint16 

BYTE = ctypes.c_ubyte
CHAR = ctypes.c_char
TCHAR = ctypes.c_char 
UCHAR = ctypes.c_ubyte

LPTSTR = ctypes.POINTER(ctypes.c_int8) 
LPCTSTR = ctypes.POINTER(ctypes.c_int8) 
LPCSTR = ctypes.POINTER(ctypes.c_int8) 
WPARAM = ctypes.c_uint32
LPARAM = ctypes.c_uint32   
LRESULT = ctypes.c_uint32
HRESULT = ctypes.c_uint32 

HWND = ctypes.c_void_p   
HGLOBAL = ctypes.c_void_p
HINSTANCE = ctypes.c_void_p
HDC = ctypes.c_void_p
HMODULE = ctypes.c_void_p
HKEY = ctypes.c_void_p
HANDLE = ctypes.c_void_p

LPBYTE = ctypes.POINTER(BYTE) 
PDWORD = ctypes.POINTER(DWORD)
PVOID = ctypes.c_void_p
PCHAR = ctypes.c_char_p
