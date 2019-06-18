#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:52:12 2019

@author: bran
"""

import numpy as np
import os

# label transform, 500-->1, 200-->2, 600-->3

data_dir = 'data'
os.chdir(data_dir)

C0_data_1ch = np.load("C0_data_1ch.npy")
C0_gt_1ch = np.load("C0_gt_1ch.npy")

LGE_data_1ch = np.load("LGE_data_1ch.npy")
LGE_gt_1ch = np.load("LGE_gt_1ch.npy")

T2_data_1ch = np.load("T2_data_1ch.npy")
T2_gt_1ch = np.load("T2_gt_1ch.npy")

test_4_data = LGE_data_1ch[71:74, ...]
test_4_gt = LGE_gt_1ch[71:74, ...]
test_4_data = test_4_data[..., np.newaxis]
test_4_gt = test_4_gt[..., np.newaxis]

test_5_data = LGE_data_1ch[74:77, ...]
test_5_gt = LGE_gt_1ch[74:77, ...]
test_5_data = test_5_data[..., np.newaxis]
test_5_gt = test_5_gt[..., np.newaxis]

train_data = np.concatenate([C0_data_1ch, LGE_data_1ch[0:71, ...], T2_data_1ch], axis = 0)
train_gt = np.concatenate([C0_gt_1ch, LGE_gt_1ch[0:71, ...], T2_gt_1ch], axis = 0)

train_data = train_data[..., np.newaxis]
train_gt = train_gt[..., np.newaxis]

np.save('train_data.npy', train_data)
np.save('train_gt.npy', train_gt)
np.save('test_4_data.npy', test_4_data)
np.save('test_4_gt.npy', test_4_gt)
np.save('test_5_data.npy', test_5_data)
np.save('test_5_gt.npy', test_5_gt)

#
#train_data = ''