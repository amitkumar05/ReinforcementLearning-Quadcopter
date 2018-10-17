#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
    DESTDIR_ARG="--root=$DESTDIR"
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/mnt/hgfs/catakin_proj/src/quad_controller_rl"

# snsure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/mnt/hgfs/catakin_proj/install/lib/python2.7/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/mnt/hgfs/catakin_proj/install/lib/python2.7/dist-packages:/mnt/hgfs/catakin_proj/build/lib/python2.7/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/mnt/hgfs/catakin_proj/build" \
    "/usr/bin/python" \
    "/mnt/hgfs/catakin_proj/src/quad_controller_rl/setup.py" \
    build --build-base "/mnt/hgfs/catakin_proj/build/quad_controller_rl" \
    install \
    $DESTDIR_ARG \
    --install-layout=deb --prefix="/mnt/hgfs/catakin_proj/install" --install-scripts="/mnt/hgfs/catakin_proj/install/bin"
