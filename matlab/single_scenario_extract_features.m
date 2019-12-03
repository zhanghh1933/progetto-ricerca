%clear all
%close all
%clc

load_functions

%% to be configured
%db_path = '../../../dataset/our_own_small_set_double_sharing/release';
db_path_m = '../dataset_complete/messenger';
db_path_t = '../dataset_complete/telegram';
db_path_w = '../dataset_complete/whatsapp';
db_path_o = '../dataset_complete/original';

M = cell(350, 1);
%M(:) = {'M'};
M(:) = {1};

T = cell(350, 1);
%T(:) = {'T'};
T(:) = {2};

W = cell(350, 1);
%W(:) = {'W'};
W(:) = {3};

O = cell(350, 1);
%O(:) = {'O'};
O(:) = {4};

target = [M; O; T; W];


%% extract additional features

[Features_M, file_path, file_name] = addi_features_recursive(db_path_m);

[Features_T, file_path, file_name] = addi_features_recursive(db_path_t);

[Features_W, file_path, file_name] = addi_features_recursive(db_path_w);

[Features_O, file_path, file_name] = addi_features_recursive(db_path_o);

features_all = [Features_M; Features_O; Features_T; Features_W];
save('../output/addi_features_all.mat', 'target' ,'features_all');



%% extract histogram of dequantized DCT coefficients
Nc = 9;         % number of AC coefficients (zig zag scan)
BT = 20;        % maximum bin value => number of bins = 2*BT+1
channel = 1;    % luminance channel

[Hist_M, file_path, file_name] = dct_coef_hist_recursive(db_path_m, channel, Nc, BT);

[Hist_T, file_path, file_name] = dct_coef_hist_recursive(db_path_t, channel, Nc, BT);

[Hist_W, file_path, file_name] = dct_coef_hist_recursive(db_path_w, channel, Nc, BT);

[Hist_O, file_path, file_name] = dct_coef_hist_recursive(db_path_o, channel, Nc, BT);

features_all = [Hist_M; Hist_O; Hist_T; Hist_W];
save('../output/deq_dct_coef_all.mat', 'target' , 'features_all');


%% extract histogram of dequantized DCT coefficients with new method
Nc = 9;         % number of AC coefficients (zig zag scan)
BT = 20;        % maximum bin value => number of bins = 2*BT+1
channel = 1;    % luminance channel


[Hist_M, file_path, file_name] = dct_coef_hist_recursive_new(db_path_m, channel, Nc, BT);

[Hist_T, file_path, file_name] = dct_coef_hist_recursive_new(db_path_t, channel, Nc, BT);

[Hist_W, file_path, file_name] = dct_coef_hist_recursive_new(db_path_w, channel, Nc, BT);

[Hist_O, file_path, file_name] = dct_coef_hist_recursive_new(db_path_o, channel, Nc, BT);

features_all = [Hist_M; Hist_O; Hist_T; Hist_W];
save('../output/new_deq_dct_coef_all.mat', 'target' , 'features_all');





