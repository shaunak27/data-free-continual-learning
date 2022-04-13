# # cancel all
# # squeue -u jsmith762 | awk '{print $1}' | tail -n+2 | xargs scancel
# rm -r _log
# mkdir _log

# Re-launch
sbatch submit_1.sh
sbatch submit_1b.sh
sbatch submit_1c.sh
sbatch submit_2.sh
sbatch submit_3.sh
sbatch submit_4.sh
sbatch submit_5.sh