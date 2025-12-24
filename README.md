# nvcuda
Version: v0.3  

Standalone version of nvcuda based from Wine-Staging  
This builds ONLY the 64bit version of nvcuda. For 32bit version  
version see:  
[https://github.com/SveSop/nvcuda/tree/32bit_only](https://github.com/SveSop/nvcuda/tree/32bit_only)  

The master branch is meant for keeping up-to-date with  
current driver implementation of nvcuda, but are a work in  
progress and not all functions are implemented.  

The DirectX implementation in the CUDA driver for linux is not  
available for this relay.  
Required driver is 525 series or newer.  
The recommended driver is always most recent. (570+).  

## Build requirements:  
- [WINE] (version >= 10.0) [https://www.winehq.org/](https://www.winehq.org/)  
- [Meson] [http://mesonbuild.com/](http://mesonbuild.com/)  
- [NINJA] [https://ninja-build.org/](https://ninja-build.org/)  
- [MINGW-W64] [https://www.mingw-w64.org/](https://www.mingw-w64.org/)

Build by running the included script:  
`./package-release.sh packagename destdir`  

## Optional test executable
If you put `--enable-tests` after the buildscript like this:  
`./package-release.sh packagename destdir --enable-tests`  
An executable will be placed in `destdir/bin` named `cudatest.exe`  

Run this using wine: `wine ./cudatest.exe` and it will perform some minor  
function tests to verify that your adapter is working with cuda.  

Building this test executable requires the build system have mingw-w64 installed.  

## Env variables
In certain cases where there is virtual or emulated hardware, nvcuda is not able to  
obtain the proper PCI ID of the adapter through wine and the call to cuDeviceGetLuid  
will fail. This can be overriden by using this env variable:  
`CUDA_FAKE_LUID=1`  
This will generate a LUID for the cuda adapter and return CUDA_SUCCESS from the call.  
PS. This should ONLY be used in those cases where this call fails due to types of  
virtual/emulated hardware as this can potentially cause other issues if enabled.  
