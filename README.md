# nvcuda
Version: v0.4  

Standalone version of nvcuda based from Wine-Staging  

This is a 32bit ONLY repo meant for older PhysX games  
that is 32bit. It will not receive updates for newer functions  
or updates as its "frozen" in this state for backward compatibility.  

CUDA SDK > 9 is dropped, as in reality there is no PhysX enabled games  
that would support this. (In fact, probably no pure 32bit PhysX games  
support > 6).  

The DirectX implementation in the CUDA driver for linux is not  
available for this relay.  
Required driver is 396 series or newer.  

## Usage
You can install 32bit nvcuda.dll into your WINEPREFIX like this:  
`WINEPREFIX=/home/myname/my-prefix ./setup_nvcuda.sh install`

You can install 32bit nvcuda.dll into PROTON like this:  
`PROTON_LIBS='/home/myname/.steam/steam/steamapps/common/Proton - Experimental' ./proton_setup.sh`  

OBS! You must have 32bit NVIDIA driver libs installed, and some games  
requires custom installation of PhysX (needs --force option for winetricks).  
Using 32bit nvcuda will ALSO mean your WINE version MUST be built as a multilib,  
and NOT using wow64 mode. Your distro MUST also have 32bit library support.  
(Same goes for PROTON version).  

## Bottles
The `bottles_setup.sh` script support installing to bottles installed from flatpak.  

## Compatibility
50xx series cards does NOT have 32bit cuda support.  

## Build requirements:  
- [WINE] (version >= 10.0) [https://www.winehq.org/](https://www.winehq.org/)  
- [Meson] [http://mesonbuild.com/](http://mesonbuild.com/)  
- [NINJA] [https://ninja-build.org/](https://ninja-build.org/)  
- [MINGW-W64] [https://www.mingw-w64.org/](https://www.mingw-w64.org/)

Build by running the included script:  
`./package-release.sh packagename destdir`  
