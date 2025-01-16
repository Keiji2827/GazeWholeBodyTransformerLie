#!/bin/bash

#python3 -m models.tools.end2end_inference --num_train_epochs 1 --seed 23 --lr 1e-5 --logging_steps 200 --n_frames 7 > output/log_231106a.log 
#echo end
#sh test_by_scene.sh

filepath=output/checkpoint-4-47436/state_dict.bin
echo $filepath
python3 -m models.tools.end2end_test --model_checkpoint $filepath 

exit

filepath=output/checkpoint-2-71155/state_dict.bin
echo $filepath
python3 -m models.tools.end2end_test --model_checkpoint $filepath 
