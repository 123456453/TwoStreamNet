import tensorflow as tf
BATCH_SIZE = 32
letters = [x for x in " abcdefghijklmnopqrstuvwxyz'?!123456789"]
initial_lr = 2e-4
step_size = 10000
gamma = 0.8
train_video_dataset = '../data/data/data/s1'
train_alignment_dataset = '../data/data/data/alignments/s1'

