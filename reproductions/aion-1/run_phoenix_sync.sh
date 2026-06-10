#!/bin/bash
# Sync phoenix's parallel-campaign cutouts into the local cache (~2x throughput).
# Files are md5(ra,dec,...)-named, so the merge is conflict-free.
LOCAL=/home2/benson/git/agentic-lensing/reproductions/aion-1/data/raw/ls_cutouts/
REMOTE=phoenix:/raid/benson/aion_campaign/ls_cutouts/
for i in $(seq 1 800); do
  rsync -a --ignore-existing -e "ssh -o BatchMode=yes" "$REMOTE" "$LOCAL" 2>/dev/null
  n=$(ls "$LOCAL" | wc -l)
  echo "$(date +%H:%M) synced; local cache now $n cutouts"
  if ssh -o BatchMode=yes phoenix 'grep -q PHOENIX_CAMPAIGN_OK /raid/benson/aion_campaign/campaign.log' 2>/dev/null; then
    rsync -a --ignore-existing -e "ssh -o BatchMode=yes" "$REMOTE" "$LOCAL" 2>/dev/null
    echo "PHOENIX_SYNC_DONE final cache $(ls "$LOCAL" | wc -l)"; break
  fi
  sleep 600
done
