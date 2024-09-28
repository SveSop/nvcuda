# nvcuda
[Bottles branch]  
This branch is a slimmer version of nvcuda meant to be used with bottles.  

## Build requirements:  
- [WINE] (version >= 9.0) [https://www.winehq.org/](https://www.winehq.org/)  
- [Meson] [http://mesonbuild.com/](http://mesonbuild.com/)  
- [NINJA] [https://ninja-build.org/](https://ninja-build.org/)  


Build by running the included script:  
`./package-release.sh destdir`  

The resulting binaries from x32 and x64 can be copied directly into the bottles  
prefix, and should run without overrides.  
This CAN be subject to changes with different runners.  

