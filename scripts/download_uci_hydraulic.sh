#!/usr/bin/env bash
# UCI "Condition Monitoring of Hydraulic Systems" (Helwig et al., 2015). Verify URL/licence before use.
set -euo pipefail
DEST=${1:-data/uci_hydraulic}
mkdir -p "$DEST"
curl -L -o "$DEST/hydraulic.zip" "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip"
unzip -o "$DEST/hydraulic.zip" -d "$DEST"
ls "$DEST"/{TS1,TS2,TS3,TS4,CE,CP,profile}.txt
