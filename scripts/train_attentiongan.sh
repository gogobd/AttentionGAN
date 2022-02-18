set -ex
# python3 train.py --dataroot ./datasets/horse2zebra --name horse2zebra_attentiongan --model attention_gan --dataset_mode unaligned --pool_size 50 --no_dropout --norm instance --lambda_A 10 --lambda_B 10 --lambda_identity 0.5 --load_size 400 --crop_size 400 --batch_size 1 --niter 60 --niter_decay 0 --gpu_ids 0 --display_id 1 --display_freq 100 --print_freq 100

python3 -m visdom.server --port 8081
python3 train.py --display_port 8081 --dataroot ./datasets/<TrainingSet> --name <TrainingSet>_attentiongan --model attention_gan --dataset_mode unaligned --pool_size 50 --no_dropout --norm instance --lambda_A 10 --lambda_B 10 --lambda_identity 0.5 --load_size 400 --crop_size 400 --batch_size 1 --niter 60 --niter_decay 0 --gpu_ids 0 --display_id 1 --display_freq 100 --print_freq 100 --save_epoch_freq 1

