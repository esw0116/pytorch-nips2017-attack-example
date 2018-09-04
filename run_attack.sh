#!/bin/bash
#
# run_attack.sh is a script which executes the attack
#
# Envoronment which runs attacks and defences calls it in a following way:
#   run_attack.sh INPUT_DIR OUTPUT_DIR MAX_EPSILON
# where:
#   INPUT_DIR - directory with input PNG images
#   OUTPUT_DIR - directory where adversarial images should be written
#   MAX_EPSILON - maximum allowed L_{\infty} norm of adversarial perturbation
#

python run_attack_iter.py \
  --input_dir="../../Dataset/NIPSAA/images" \
  --output_dir="./outputs" \
  --max_epsilon="10" \
  --checkpoint_path="./inception_v3_google-1a9a5a14.pth"

