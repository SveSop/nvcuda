#!/bin/bash

set -e

shopt -s extglob

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 releasename destdir [--enable-tests]"
  exit 1
fi

VERSION="$1"
NVCUDA_SRC_DIR=$(dirname "$(readlink -f "$0")")
NVCUDA_BUILD_DIR=$(realpath "$2")"/nvcuda-$VERSION"
ENABLE_TESTS=false

if [ $# -eq 3 ]; then
  case "$3" in
    --enable-tests)
      ENABLE_TESTS=true
      ;;
    *)
      echo "Error: unknown option '$3'"
      echo "Usage: $0 releasename destdir [--enable-tests]"
      exit 1
      ;;
  esac
fi

if [ -e "$NVCUDA_BUILD_DIR" ]; then
  echo "Build directory $NVCUDA_BUILD_DIR already exists"
  exit 1
fi

# build nvcuda

cd "$NVCUDA_SRC_DIR"

meson setup                                            \
      --cross-file "$NVCUDA_SRC_DIR/build-wine64.txt"  \
      --buildtype release                              \
      --prefix "$NVCUDA_BUILD_DIR"                     \
      --libdir x64                                     \
      --strip                                          \
      "$NVCUDA_BUILD_DIR/build.64"

cd "$NVCUDA_BUILD_DIR/build.64"
ninja install

rm -R "$NVCUDA_BUILD_DIR/build.64"

# Optional cudatest.exe build
if $ENABLE_TESTS; then

  cd "$NVCUDA_SRC_DIR"
  meson setup                                              \
      --cross-file "$NVCUDA_SRC_DIR/tests/build-win64.txt" \
      --buildtype release                                  \
      --prefix "$NVCUDA_BUILD_DIR"                         \
      --libdir bin                                         \
      --strip                                              \
      "$NVCUDA_BUILD_DIR/build.tests"                      \

  cd "$NVCUDA_BUILD_DIR/build.tests"
  ninja install

  rm -R "$NVCUDA_BUILD_DIR/build.tests"
fi

# cleanup
cd $NVCUDA_BUILD_DIR
find . -name \*.a -type f -delete
find . -name '*.dll.so' -type f -exec sh -c 'mv "$1" "${1%.so}"' _ {} \;
echo "Done building!"
