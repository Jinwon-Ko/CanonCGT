set -Eeuo pipefail
trap 'echo "[ERROR] line $LINENO: command failed. Exit."; exit 1' ERR

 # STAGE 1
 echo "[INFO] <STAGE1> Start..."
 echo "[INFO] <STAGE1> {style encoder} Training Start ..."
 python main.py --gpu 0 --yaml Stage1_style_encoder --host server --run_mode train
 echo "[INFO] <STAGE1> {style encoder} Training Finished."

 echo "[INFO] <STAGE1> {style encoder} Testing Start (Save style centroids)..."
 python main.py --gpu 0 --yaml Stage1_style_encoder --host server --run_mode test --load
 echo "[INFO] <STAGE1> {style encoder} Testing Finished."

 sleep 3

 echo "[INFO] <STAGE1> {canonicalizer & styler} Training Start..."
 python main.py --gpu 0 --yaml Stage1_canonicalizer --host server --run_mode train &
 python main.py --gpu 1 --yaml Stage1_styler --host server --run_mode train &
 wait
 echo "[INFO] <STAGE1> {canonicalizer & styler} Training Finished."
 echo "[INFO] <STAGE1> End..."


 # STAGE 2
 sleep 3
 echo "[INFO] <STAGE2> Start..."
 echo "[INFO] <STAGE2> Testing Start from <STAGE1>..."
 python main.py --gpu 0 --yaml Stage2_end_to_end_finetuning --host server --run_mode test
 echo "[INFO] <STAGE2> Testing Finished from <STAGE1>."


 echo "[INFO] <STAGE2> {finetuning} Training Start..."
 python main.py --gpu 0 --yaml Stage2_end_to_end_finetuning --host server --run_mode train
 echo "[INFO] <STAGE2> {finetuning} Training Finished}."


 echo "[INFO] <STAGE2> {finetuning} Testing Start from <STAGE2>..."
 python main.py --gpu 0 --yaml Stage2_end_to_end_finetuning --host server --run_mode test --load
 echo "[INFO] <STAGE2> {finetuning} Testing Finished from <STAGE2>."
 echo "[INFO] <STAGE2> End..."

# STAGE 3
sleep 3
echo "[INFO] <STAGE3> Start..."
echo "[INFO] <STAGE3> Testing Start from <STAGE2>..."
python main.py --gpu 0 --yaml Stage3_SSL_training_Flickr2K_PPR10K_LSDIR --host server --run_mode test
echo "[INFO] <STAGE3> Testing Finished from <STAGE2>."

echo "[INFO] <STAGE3> {extra-trained model} Training Start..."
python main.py --gpu 0 --yaml Stage3_SSL_training_Flickr2K_PPR10K_LSDIR --host server --run_mode train
echo "[INFO] <STAGE3> {extra-trained model} Training Finished."


echo "[INFO] <STAGE3> {extra-trained model} Testing Start..."
python main.py --gpu 0 --yaml Stage3_SSL_training_Flickr2K_PPR10K_LSDIR --host server --run_mode test --load
echo "[INFO] <STAGE3> {extra-trained model} Testing Finished."

echo "[INFO] All Done!"
