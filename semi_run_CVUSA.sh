#!/usr/bin/env bash

sbatch -p gpu --gres=gpu:1 --cpus-per-gpu=8 --mem-per-cpu=8G -C "gmem48" --job-name="part" --gres-flags=enforce-binding --exclude c4-5 --exclude c4-4\
 --output="jobs/semi_per20_mse_loss_no_mining_sampler_unsup_wt_10.out"\
 --wrap="python -u semi_train.py --lr 0.0001 --batch-size 32 -wt_ul 1.0\
 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed\
 --world-size 1 --rank 0  --epochs 100 --save_path ./log_train_wts/semi_per20_mse_loss_no_mining_sampler_unsup_wt_10 --op sam\
 --wd 0.03 --dataset cvusa --cos --dim 1000 --asam --rho 2.5\
 -l train-19zl_20%_1.csv
 -ul train-19zl_80%_1.csv"

sleep 10

sbatch -p gpu --gres=gpu:1 --cpus-per-gpu=8 --mem-per-cpu=8G -C "gmem48" --job-name="part" --gres-flags=enforce-binding --exclude c4-5 --exclude c4-4\
 --output="jobs/semi_per20_mse_loss_no_mining_sampler_unsup_wt_05.out"\
 --wrap="python -u semi_train.py --lr 0.0001 --batch-size 32 -wt_ul 0.5\
 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed\
 --world-size 1 --rank 0  --epochs 100 --save_path ./log_train_wts/semi_per20_mse_loss_no_mining_sampler_unsup_wt_05 --op sam\
 --wd 0.03 --dataset cvusa --cos --dim 1000 --asam --rho 2.5\
 -l train-19zl_20%_1.csv
 -ul train-19zl_80%_1.csv"

sleep 10

sbatch -p gpu --gres=gpu:1 --cpus-per-gpu=8 --mem-per-cpu=8G -C "gmem48" --job-name="part" --gres-flags=enforce-binding --exclude c4-5 --exclude c4-4\
 --output="jobs/semi_per20_mse_loss_no_mining_sampler_unsup_wt_01.out"\
 --wrap="python -u semi_train.py --lr 0.0001 --batch-size 32 -wt_ul 0.1\
 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed\
 --world-size 1 --rank 0  --epochs 100 --save_path ./log_train_wts/semi_per20_mse_loss_no_mining_sampler_unsup_wt_01 --op sam\
 --wd 0.03 --dataset cvusa --cos --dim 1000 --asam --rho 2.5\
 -l train-19zl_20%_1.csv
 -ul train-19zl_80%_1.csv"


sleep 10

sbatch -p gpu --gres=gpu:1 --cpus-per-gpu=8 --mem-per-cpu=8G -C "gmem48" --job-name="part" --gres-flags=enforce-binding --exclude c4-5 --exclude c4-4\
 --output="jobs/semi_per20_mse_loss_no_mining_sampler_unsup_wt_03.out"\
 --wrap="python -u semi_train.py --lr 0.0001 --batch-size 32 -wt_ul 0.3\
 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed\
 --world-size 1 --rank 0  --epochs 100 --save_path ./log_train_wts/semi_per20_mse_loss_no_mining_sampler_unsup_wt_03 --op sam\
 --wd 0.03 --dataset cvusa --cos --dim 1000 --asam --rho 2.5\
 -l train-19zl_20%_1.csv
 -ul train-19zl_80%_1.csv"


sleep 10

sbatch -p gpu --gres=gpu:1 --cpus-per-gpu=8 --mem-per-cpu=8G -C "gmem48" --job-name="part" --gres-flags=enforce-binding --exclude c4-5 --exclude c4-4\
 --output="jobs/semi_per20_mse_loss_no_mining_sampler_unsup_wt_07.out"\
 --wrap="python -u semi_train.py --lr 0.0001 --batch-size 32 -wt_ul 0.7\
 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed\
 --world-size 1 --rank 0  --epochs 100 --save_path ./log_train_wts/semi_per20_mse_loss_no_mining_sampler_unsup_wt_07 --op sam\
 --wd 0.03 --dataset cvusa --cos --dim 1000 --asam --rho 2.5\
 -l train-19zl_20%_1.csv
 -ul train-19zl_80%_1.csv"

sleep 10

sbatch -p gpu --gres=gpu:1 --cpus-per-gpu=8 --mem-per-cpu=8G -C "gmem48" --job-name="part" --gres-flags=enforce-binding --exclude c4-5 --exclude c4-4\
 --output="jobs/semi_per10_mse_loss_no_mining_sampler_unsup_wt_10.out"\
 --wrap="python -u semi_train.py --lr 0.0001 --batch-size 32 -wt_ul 1.0\
 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed\
 --world-size 1 --rank 0  --epochs 100 --save_path ./log_train_wts/semi_per10_mse_loss_no_mining_sampler_unsup_wt_10 --op sam\
 --wd 0.03 --dataset cvusa --cos --dim 1000 --asam --rho 2.5\
 -l train-19zl_10%_1.csv
 -ul train-19zl_90%_1.csv"

