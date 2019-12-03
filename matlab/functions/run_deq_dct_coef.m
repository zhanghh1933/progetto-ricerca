clear all
close all
clc

addpath('../libs/jpeg_toolbox');
addpath('../libs/ZIG_ZIG_SCAN');

% to be configured
db_path = '../../dataset/our_own_small_set_double_sharing/release';

%% extract histogram of dequantized DCT coefficients
Nc = 9;         % number of AC coefficients (zig zag scan)
BT = 20;        % maximum bin value => number of bins = 2*BT+1
channel = 1;    % luminance channel
[Hist, file_path, file_name] = dct_coef_hist_recursive(db_path, channel, Nc, BT);
save('../../output/deq_dct_coef.mat', 'Hist', 'file_path', 'file_name');
