#!/bin/bash

nvcuda32_dir="$(dirname "$(readlink -fm "$0")")"
bottles_dir="$HOME/.var/app/com.usebottles.bottles/data/bottles/bottles"
win='drive_c/windows/syswow64'
lib='x32'

if [ ! -f "$nvcuda32_dir/$lib/nvcuda.dll" ]; then
    echo "Files not found in $nvcuda32_dir/$lib" >&2
    exit 1
fi

if [ -z "$1" ]; then
    echo -ne "BOTTLE is not set!\n"
    echo -ne "This is a list of your available bottles:\n"
    echo -ne "\n=========================================\n"
    ls -1 $bottles_dir
    echo -ne "=========================================\n\n"
    echo -ne "Specify your bottle. Eg: ./bottles-install.sh MyBottle\n"
    exit 1
fi

if [ ! -f "$bottles_dir/$1/$win/dxgi.dll" ]; then
    echo -ne "Windows files not found in $bottles_dir\$1! check bottle installation\n" >&2
    exit 1
else
    BOTTLE="$1"
fi

cp -f "$nvcuda32_dir/$lib/nvcuda.dll" "$bottles_dir/$BOTTLE/$win/"

echo -ne "All done - 32bit NVCUDA library copied to $BOTTLE\n"
