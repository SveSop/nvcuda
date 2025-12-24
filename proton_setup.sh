#!/bin/bash

nvlibs_dir="$(dirname "$(readlink -fm "$0")")"
lib='lib/wine'
arch='i386'

if [ ! -f "$nvlibs_dir/x32/nvcuda.dll" ]; then
    echo "Files not found in $nvlibs_dir" >&2
    exit 1
fi

if [ -z "$PROTON_LIBS" ]; then
    echo -ne "PROTON_LIBS is not set!\n"
    echo -ne "Example: PROTON_LIBS='$HOME/.steam/steam/steamapps/common/Proton - Experimental'\n"
    exit 1
else
    PROTON_LIBS="$(readlink -fm "$PROTON_LIBS")"
fi

if [ ! -f "$PROTON_LIBS/files/$lib/$arch-windows/dxgi.dll" ]; then
    echo -ne "Proton files not found in $PROTON_LIBS! Proton not installed or wrong path!\n" >&2
    exit 1
fi

# proton 10/Experimental support
cp -f "$nvlibs_dir/x32/nvcuda.dll" "$PROTON_LIBS/files/$lib/$arch-windows/nvcuda.dll"

echo -ne "All done - Files dropped in $PROTON_LIBS\n"
