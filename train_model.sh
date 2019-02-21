#!/bin/bash

python train.py \
--trial 206 \
--model VRNN_MIXED \
--dataset boids \
--n_agents 8 \
--x_dim 2 \
--y_dim 16 \
--m_dim 90 \
--z_dim 2 \
--h_dim 32 \
--rnn_dim 200 \
--rnn_macro_dim 200 \
--rnn_micro_dim 200 \
--n_layers 2 \
--subsample 1 \
--start_lr 1e-4 \
--min_lr 1e-4 \
--n_epochs 200 \
--save_every 50 \
--batch_size 512 \
--seed 200 \
--cuda
