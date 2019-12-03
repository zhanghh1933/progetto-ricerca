%clear all
%close all
%clc

%addpath('../../libs/jpeg_toolbox');

%% to be configured
%db_path = '../../../dataset/our_own_small_set_double_sharing/release';
db_path_m_m = '../double_sharing/messenger/messenger';
db_path_m_t = '../double_sharing/messenger/telegram';
db_path_m_w = '../double_sharing/messenger/whatsapp';

db_path_t_m = '../double_sharing/telegram/messenger';
db_path_t_t = '../double_sharing/telegram/telegram';
db_path_t_w = '../double_sharing/telegram/whatsapp';


db_path_w_m = '../double_sharing/whatsapp/messenger';
db_path_w_t = '../double_sharing/whatsapp/telegram';
db_path_w_w = '../double_sharing/whatsapp/whatsapp';

db_path_o = '../dataset_complete/original';


m_m = cell(350, 1);
m_m(:) = {11};
m_t = cell(350, 1);
m_t(:) = {12};
m_w = cell(350, 1);
m_w(:) = {13};

t_m = cell(350, 1);
t_m(:) = {21};
t_t = cell(350, 1);
t_t(:) = {22};
t_w = cell(350, 1);
t_w(:) = {23};

w_m = cell(350, 1);
w_m(:) = {31};
w_t = cell(350, 1);
w_t(:) = {32};
w_w = cell(350, 1);
w_w(:) = {33};

o = cell(350, 1);
o(:) = {4};

target = [m_m; m_t; m_w;  t_m; t_t; t_w; w_m; w_t; w_w; o];


%% extract additional features
 
[Features_m_m, file_path, file_name] = addi_features_recursive(db_path_m_m);
[Features_m_t, file_path, file_name] = addi_features_recursive(db_path_m_t);
[Features_m_w, file_path, file_name] = addi_features_recursive(db_path_m_w);

[Features_t_m, file_path, file_name] = addi_features_recursive(db_path_t_m);
[Features_t_t, file_path, file_name] = addi_features_recursive(db_path_t_t);
[Features_t_w, file_path, file_name] = addi_features_recursive(db_path_t_w);

[Features_w_m, file_path, file_name] = addi_features_recursive(db_path_w_m);
[Features_w_t, file_path, file_name] = addi_features_recursive(db_path_w_t);
[Features_w_w, file_path, file_name] = addi_features_recursive(db_path_w_w);

[Features_o, file_path, file_name] = addi_features_recursive(db_path_o);

features_all = [Features_m_m; Features_m_t; Features_m_w; 
    Features_t_m; Features_t_t; Features_t_w;
    Features_w_m; Features_w_t; Features_w_w;
    Features_o];

save('outputNew/addi_features_all_double.mat', 'target' ,'features_all');



%% extract histogram of dequantized DCT coefficients
Nc = 9;         % number of AC coefficients (zig zag scan)
BT = 20;        % maximum bin value => number of bins = 2*BT+1
channel = 1;    % luminance channel

[Hist_m_m, file_path, file_name] = dct_coef_hist_recursive_new(db_path_m_m, channel, Nc, BT);
[Hist_m_t, file_path, file_name] = dct_coef_hist_recursive_new(db_path_m_t, channel, Nc, BT);
[Hist_m_w, file_path, file_name] = dct_coef_hist_recursive_new(db_path_m_w, channel, Nc, BT);

[Hist_t_m, file_path, file_name] = dct_coef_hist_recursive_new(db_path_t_m, channel, Nc, BT);
[Hist_t_t, file_path, file_name] = dct_coef_hist_recursive_new(db_path_t_t, channel, Nc, BT);
[Hist_t_w, file_path, file_name] = dct_coef_hist_recursive_new(db_path_t_w, channel, Nc, BT);

[Hist_w_m, file_path, file_name] = dct_coef_hist_recursive_new(db_path_w_m, channel, Nc, BT);
[Hist_w_t, file_path, file_name] = dct_coef_hist_recursive_new(db_path_w_t, channel, Nc, BT);
[Hist_w_w, file_path, file_name] = dct_coef_hist_recursive_new(db_path_w_w, channel, Nc, BT);

[Hist_o, file_path, file_name] = dct_coef_hist_recursive_new(db_path_o, channel, Nc, BT);

features_all = [
    Hist_m_m; Hist_m_t; Hist_m_w; 
    Hist_t_m; Hist_t_t; Hist_t_w;
    Hist_w_m; Hist_w_t; Hist_w_w;
    Hist_o];



%cambiare il nome del file 
save('outputNew/deq_dct_coef_all_double.mat', 'target' , 'features_all');



