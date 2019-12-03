clear all
close all
clc

addpath('../../libs/jpeg_toolbox');

% to be configured
db_path = '../../../dataset/our_own_small_set_double_sharing/release';

%% extract additional features

[Features, file_path, file_name] = addi_features_recursive(db_path);
save('../../output/addi_features.mat', 'Features', 'file_path', 'file_name');
