function [Hist, file_path, file_name] = dct_coef_hist_recursive_new(db_path, channel, Nc, BT)
%   This function extracts DCT coefficient histograms of the whole dataset
%   Input:
%       db_path: the root path of the dataset
%       channel: 1 (luminance), 2 (chrominance)
%       Nc: number of analyzed DCT coeff in each 8x8 blocks
%       BT: the limit of histogram ==> number of bins = 2*BT + 1
%   Output:
%       Hist: the nxd histogram matrix; n is the number of images; d is the
%       number of histogram bins
%       file_path: a list containing file paths (used to split data later)
%       file_name: a list containing file names


    db_path = GetFullPath(db_path);
    [file_path, file_name] = get_file_list(db_path, [], []);
    N = length(file_path);
    fprintf('\nNumber of images: %d \n', N);
    
    Hist = zeros(N, ((2*BT+1)*Nc));
    for i = 1:N
        fprintf('process file: %s\n', char(file_name(i)));
        %jobj=jpeg_read(char(file_path(i)));
        jobj = imread(char(file_path(i)));
        hist = dct_hist_extraction(jobj, Nc, BT);
        Hist(i, :) = reshape(hist, 1, []);
    end
end
