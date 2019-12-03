function [Features, file_path, file_name] = addi_features_recursive(db_path)
%	This function extracts all additional features from a dataset
%   Input:
%       db_path: the root path of the dataset
%   Output:
%       Features: the nxd matrix; n is the number of images; d is the
%       dimension of the feature vector
    db_path = GetFullPath(db_path);
    [file_path, file_name] = get_file_list(db_path, [], []);
    N = length(file_path);
    fprintf('\nNumber of images: %d \n', N);
    Features = zeros(N, 152);
    for i = 1:N
        fprintf('process file: %s\n', char(file_name(i)));
        jobj=jpeg_read(char(file_path(i)));
        Features(i,:) = addi_features(jobj);
    end
    
end

