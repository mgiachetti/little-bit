#!/bin/bash
# Poll CloudRift until a GPU we can use is free, rent ONE VM, notify, exit.
#   scripts/cloudrift_grab.sh [interval_s] [name]
# Types are tried in priority order. Excludes the Seoul RTX 5090s (GSP passthrough bug).
INTERVAL=${1:-300}
NAME=${2:-little-bit-bench}
IMAGE=https://storage.googleapis.com/cloudrift-vm-disks/disks/github/ubuntu-noble-server-gpu-580-129-20260716-134310.img
CLOUDINIT=https://storage.googleapis.com/cloudrift-vm-disks/cloudinit/ubuntu-base.cloudinit
WANTED="l40s-24c-kn rtxpro6000-11-50-500-1l rtxpro6000-12-100-1500-nr rtxpro6000-4-100-1000-ti rtxpro6000mq-11-56-850-1lg a100-16-210-800-generic rtx49-7c-kn rtx49-10c-kn rtx49-7-50-500-nr rtx49-15-80-400-ec rtx49-18-80-450-kn-packed"
LOG=${TMPDIR:-/tmp}/cloudrift_grab.log

notify() { osascript -e "display notification \"$1\" with title \"CloudRift\" sound name \"Glass\"" 2>/dev/null; echo "$(date '+%F %T') $1" | tee -a "$LOG"; }

while true; do
  list=$(rift instance-type list --service vm 2>/dev/null)
  for t in $WANTED; do
    avail=$(echo "$list" | awk -v t="$t" '$1 == t {print $(NF-1)}')
    if [ -n "$avail" ] && [ "$avail" -gt 0 ]; then
      echo "$(date '+%F %T') $t available ($avail), renting" | tee -a "$LOG"
      out=$(rift instance rent --instance-type "$t.1" --name "$NAME" --ssh-key mac-ed25519 \
              --image "$IMAGE" --cloudinit-url "$CLOUDINIT" 2>&1)
      if echo "$out" | grep -qE '^[0-9a-f]{8}-'; then
        notify "Rented $t ($NAME): $out"
        exit 0
      fi
      echo "$(date '+%F %T') rent of $t failed: $out" | tee -a "$LOG"
    fi
  done
  echo "$(date '+%F %T') nothing yet" >> "$LOG"
  sleep "$INTERVAL"
done
