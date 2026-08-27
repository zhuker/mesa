#!/bin/bash
# Names every compute app on the card once a second, so a foreign tenant is
# recorded rather than missed between runs.
while true; do
  echo "$(date +%s) $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ';')"
  sleep 1
done
