#!/bin/bash
# Sequenced autonomous driver:
#  1) wait flagship done + ddpayne join done -> main GPU batch (tasks 3,2,10,6,11)
#  2) wait for gz10 cutouts (full OR a deadline) -> gz10 GPU batch (tasks 4,7,8)
# Sequencing avoids GPU contention; the main batch is never blocked by the slow
# gz10 cutout fetch, and the gz10 batch runs on whatever cutouts are cached by
# its deadline.
cd /home2/benson/git/agentic-lensing/reproductions/aion-1
until grep -q IMAGE_CONFIGS_DONE data/results/_image_configs.log 2>/dev/null; do sleep 30; done
until grep -q DDPAYNE_DESI_OK data/results/_ddpayne_desi.log 2>/dev/null; do sleep 15; done
echo "WATCH: flagship+ddpayne ready @ $(date) -> main batch"
bash run_main_gpu.sh

# wait for gz10 full completion or a 45-min deadline, whichever first
deadline=$((SECONDS + 2700))
until grep -q GZ10_IMAGES_OK data/results/_fetch_gz10.log 2>/dev/null || [ $SECONDS -ge $deadline ]; do
  sleep 60
done
echo "WATCH: gz10 ready/deadline @ $(date) -> gz10 batch"
bash run_gz10_gpu.sh
echo "WATCH_ALL_DONE @ $(date)"
