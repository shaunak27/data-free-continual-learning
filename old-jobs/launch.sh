# cancel all
# squeue -u jsmith762 | awk '{print $1}' | tail -n+2 | xargs scancel


# Re-launch
sbatch submit_1.sh
sbatch submit_2.sh
sbatch submit_3.sh
sbatch submit_4.sh








# Close to deadline
# # # 80 - 5
# # sbatch _jobs3/submit_a.sh
# # sbatch _jobs3/submit_b.sh
# # sbatch _jobs3/submit_c.sh
# # sbatch _jobs3/submit_d.sh
# # sbatch _jobs3/submit_e.sh

# # # 10 - 10
# # sbatch _jobs/submit_b.sh
# # sbatch _jobs/submit_c.sh
# # sbatch _jobs/submit_d.sh
# # sbatch _jobs/submit_e.sh
# # sbatch _jobs/submit_a.sh

# # # higher priority
# # sbatch _jobsc/submit_a.sh
# # sbatch _jobs2/submit_a.sh

# # # 95 - 5
# # sbatch _jobs2/submit_b.sh
# # sbatch _jobs2/submit_c.sh
# # sbatch _jobs2/submit_d.sh
# # sbatch _jobs2/submit_e.sh

# # # 10 - 10 - pt
# # sbatch _jobsc/submit_b.sh
# # sbatch _jobsc/submit_c.sh
# # sbatch _jobsc/submit_d.sh
# # sbatch _jobsc/submit_e.sh