#!/bin/bash

set -e

shopt -s extglob

if [ -z "$1" ]; then
  echo "Usage: package-release.sh destdir"
  exit 1
fi

NVCUDA_SRC_DIR=`dirname $(readlink -f $0)`
NVCUDA_BUILD_DIR=$(realpath "$1")"/nvcuda"

if [ -e "$NVCUDA_BUILD_DIR" ]; then
  echo "Build directory $NVCUDA_BUILD_DIR already exists"
  exit 1
fi

# build nvcuda

function build_arch {
  export WINEARCH="win$1"

  cd "$NVCUDA_SRC_DIR"

  meson --cross-file "$NVCUDA_SRC_DIR/build-wine$1.txt"  \
        --buildtype release                              \
        --prefix "$NVCUDA_BUILD_DIR"                     \
        --libdir lib$1                                   \
	--strip                                          \
        "$NVCUDA_BUILD_DIR/build.$1"

  cd "$NVCUDA_BUILD_DIR/build.$1"
  ninja install

  rm -R "$NVCUDA_BUILD_DIR/build.$1"
}

build_arch 64
build_arch 32

# cleanup
cd $NVCUDA_BUILD_DIR
find . -name \*.a -type f -delete
mv lib32 lib
echo "Done building!"
