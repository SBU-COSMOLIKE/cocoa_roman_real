if [ -z "${ROOTDIR}" ]; then
    echo 'ERROR ENV VARIABLE ROOTDIR IS NOT DEFINED' >&2
    return 1
fi
if [ -z "${CXX_COMPILER}" ]; then
    echo 'ERROR ENV VARIABLE CXX_COMPILER IS NOT DEFINED' >&2
    return 1
fi
if [ -z "${C_COMPILER}" ]; then
    echo 'ERROR ENV VARIABLE C_COMPILER IS NOT DEFINED' >&2
    return 1
fi
if [ -z "${MAKE_NUM_THREADS}" ]; then
    echo 'ERROR ENV VARIABLE MAKE_NUM_THREADS IS NOT DEFINED' >&2
    return
fi

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------- COMPILE COSMOLIKE ----------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

cd $ROOTDIR/projects/lsst_y1/interface

make -f MakefileCosmolike clean

make -j $MAKE_NUM_THREADS -f MakefileCosmolike all

cd $ROOTDIR

