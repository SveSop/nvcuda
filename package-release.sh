#!/bin/bash

set -e

shopt -s extglob

if [ $# -lt 2 ]; then
  echo "Usage: $0 releasename destdir [--enable-tests] [--fakedll]"
  exit 1
fi

VERSION="$1"
DESTDIR="$2"
NVCUDA_SRC_DIR=$(dirname "$(readlink -f "$0")")
NVCUDA_BUILD_DIR=$(realpath "$2")"/nvcuda-$VERSION"
shift 2

ENABLE_TESTS=false
FAKEDLL=""
LIBDIR='x64'

for arg in "$@"; do
  case "$arg" in
    --enable-tests)
      ENABLE_TESTS=true
      ;;
    --fakedll)
      FAKEDLL="-Dfakedll=true"
      LIBDIR='lib'
      ;;
    *)
      echo "Error: unknown option '$arg'"
      exit 1
      ;;
  esac
done

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
      --libdir $LIBDIR                                 \
      --strip                                          \
      $FAKEDLL                                         \
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
if [ -z "$FAKEDLL" ]; then
  find . -name '*.dll.so' -type f -exec sh -c 'mv "$1" "${1%.so}"' _ {} \;
fi
echo "Done building!"
