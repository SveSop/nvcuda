#!/bin/bash

nvcuda_dir="$(dirname "$(readlink -fm "$0")")"
lib='x32'

if [ ! -f "$nvcuda_dir/$lib/nvcuda.dll" ]; then
    echo "Files not found in $nvcuda_dir/$lib" >&2
    exit 1
fi

if [ -z "$WINEPREFIX" ]; then
    echo -ne "WINEPREFIX is not set! This may create a wineprefix in the default distro folder)\nContinue? (Y/N)\n"
    old_stty_cfg=$(stty -g)
    stty raw -echo ; answer=$(head -c 1) ; stty $old_stty_cfg
    if echo "$answer" |grep -iq "^y" ;then
        wineboot -u
    else
        exit 1
    fi
else
    if ! [ -f "$WINEPREFIX/system.reg" ]; then
        echo -ne "WINEPREFIX does not point to an existing wine installation.\nProceeding will create a new one in $WINEPREFIX\nContinue? (Y/N)\n"
        old_stty_cfg=$(stty -g)
        stty raw -echo ; answer=$(head -c 1) ; stty $old_stty_cfg
        if echo "$answer" |grep -iq "^y" ;then
            wineboot -u
        else
            exit 1
        fi
    fi
fi

function remove {
    echo "    Removing nvcuda.dll..."
    local dll="$WINEPREFIX/drive_c/windows/syswow64/nvcuda.dll"
    if [ -e "$dll" ]; then
        out=$(rm "$dll" 2>&1)
        if [ $? -ne 0 ]; then
            echo -e "$out"
            exit=2
        fi
        echo -e "    nvcuda.dll removed from $WINEPREFIX"
    else
        echo -e "    '$dll' does not exist!"
        exit=2
    fi
}

function install {
    echo "    Installing nvcuda.dll..."
    cp -f "$nvcuda_dir/$lib/nvcuda.dll" "$WINEPREFIX/drive_c/windows/syswow64"
    if [ $? -ne 0 ]; then
        echo -e "    Failed to install nvcuda.dll!"
        exit 2
    else
        echo "    'nvcuda.dll' copied to: $WINEPREFIX."
    fi
}

case "$1" in
uninstall)
    fun=remove
    ;;
install)
    fun=install
    ;;
*)
    echo "Unrecognized option: $1"
    echo "Usage: $0 [install|uninstall]"
    exit 1
    ;;
esac

$fun

