# nvcuda
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


Build by running the included script:  
`./package-release.sh destdir`  
