#!/bin/bash
function echo_and_run { echo -e "\$ $@" ; read input; "$@" ; read input; }
echo "********** EVO Testing ************"
echo "evo on sequence $2"
cd /home/phy12321/code/Multi_view_Cylinder_Net/Logs/"${1}" || exit
ref_pose="/home/phy12321/code/Multi_view_Cylinder_Net/dataset/kitti/poses/${2}.txt"
pred_pose="/home/phy12321/code/Multi_view_Cylinder_Net/Logs/${1}/results/pred_pose_${2}.txt"
save_results="/home/phy12321/code/Multi_view_Cylinder_Net/Logs/${1}/results/evo_ape_${2}.zip"
save_plot="/home/phy12321/code/Multi_view_Cylinder_Net/Logs/${1}/results/ape_${2}.pdf"
echo -e "y\ny\n" | echo_and_run evo_ape kitti "${ref_pose}" "${pred_pose}"  -r full --save_results "${save_results}" -vas --plot_mode xz --save_plot "${save_plot}"

ref_pose="/home/phy12321/code/Multi_view_Cylinder_Net/dataset/kitti/poses/${2}.txt"
pred_pose="/home/phy12321/code/Multi_view_Cylinder_Net/Logs/${1}/results/pred_pose_${2}.txt"
save_results="/home/phy12321/code/Multi_view_Cylinder_Net/Logs/${1}/results/evo_rpe_${2}.zip"
save_plot="/home/phy12321/code/Multi_view_Cylinder_Net/Logs/${1}/results/rpe_${2}.pdf"
echo -e "y\ny\n" | echo_and_run evo_rpe kitti "${ref_pose}" "${pred_pose}"  -r trans_part --delta 100 --save_results "${save_results}" -vas --plot_mode xz --save_plot "${save_plot}"
