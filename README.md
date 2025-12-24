# nvcuda
Version: v0.2  

Standalone version of nvcuda based from Wine-Staging  

This is a 32bit ONLY repo meant for older PhysX games  
that is 32bit. It will not receive updates for newer functions  
or updates as its "frozen" in this state for backward compatibility.  

I may remove SDK 12/13 support to lighten the binary a bit, as this  
is not really supported by 32bit anyway, and no apps or games that  
are 32bit would ever require that.  

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
