# nvcuda
Version: v0.2  

Standalone version of nvcuda based from Wine-Staging  

The master branch is meant for keeping up-to-date with  
current driver implementation of nvcuda, but are a work in  
progress and not all functions are implemented.  

The DirectX implementation in the CUDA driver for linux is not  
available for this relay.  
Required driver is 525 series or newer.  
The recommended driver is always most recent. (570+).  

## Build requirements:  
- [WINE] (version >= 9.0) [https://www.winehq.org/](https://www.winehq.org/)  
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
