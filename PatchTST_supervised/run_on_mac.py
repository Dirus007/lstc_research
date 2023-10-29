import subprocess

# define the command

cmd = """
/opt/homebrew/bin/python3 -u run_longExp.py \
  --random_seed 2021 \
  --is_training 1 \
  --root_path ./dataset/weather \
  --data_path weather.csv \
  --model_id weather_336_96 \
  --model PatchTST \
  --data custom \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --enc_in 21 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 128 \
  --d_ff 256 \
  --dropout 0.2 \
  --fc_dropout 0.2 \
  --head_dropout 0 \
  --patch_len 16 \
  --stride 8 \
  --des 'Exp' \
  --train_epochs 1 \
  --patience 20 \
  --experiment_name 'Dim_mixing_patch_with_vanilla_trasformer'\
  --log_to_wandb \
  --itr 1 --batch_size 128 --learning_rate 0.0001 > logs/LongForecasting/PatchTST_weather_336_96.log
"""

# ensure the logs directory exists
subprocess.run(["mkdir", "-p", "./logs/LongForecasting"], check=True)

# run the command
subprocess.run(cmd, shell=True, check=True)
