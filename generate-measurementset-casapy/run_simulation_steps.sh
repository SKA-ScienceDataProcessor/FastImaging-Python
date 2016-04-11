#!/usr/bin/env bash

CASAPY="casapy --nogui"

$CASAPY -c "generate-ms.py"
$CASAPY -c "export-uvw.py"
cp -r "vla-sim.MS" "vla-resim.MS"
python regenerate-vis.py
$CASAPY -c "reimport-data-to-casa.py"
$CASAPY -c "image-ms.py"