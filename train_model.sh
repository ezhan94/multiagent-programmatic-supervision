#!/bin/bash

python train.py \
--trial 201 \
--model VRNN_SINGLE \
--dataset bball \
--n_agents 5 \
--x_dim 2 \
--y_dim 10 \
--m_dim 90 \
--z_dim 80 \
--h_dim 200 \
--rnn_dim 900 \
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
