#!/usr/bin/env bash


CODE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"
CASAPY="casa --nogui"
OUTDIR="./simulation_output"

mkdir -p $OUTDIR
cp ${CODE_DIR}/vla.c.cfg $OUTDIR
cd $OUTDIR

$CASAPY -c "${CODE_DIR}/generate-ms.py"
$CASAPY -c "${CODE_DIR}/export-uvw.py"
cp -r "vla-sim.MS" "vla-resim.MS"
python "${CODE_DIR}/regenerate-vis.py"
$CASAPY -c "${CODE_DIR}/reimport-data-to-casa.py"
$CASAPY -c "${CODE_DIR}/image-ms.py"


fastimg_sourcefind vla.image.fits