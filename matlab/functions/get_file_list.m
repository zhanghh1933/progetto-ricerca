function [file_path, file_name] = get_file_list(path, file_path, file_name)
    list_dir = dir(path);
    for f = list_dir'
        if strcmp(f.name, '.') || strcmp(f.name, '..') || strcmp(f.name, '.DS_Store') || ...
                strcmp(f.name, '._.DS_Store') || ~isempty(strfind(f.name, '._'))
            continue
        end
        if f.isdir == 1
            [file_path, file_name] = get_file_list([path filesep f.name], file_path, file_name);
        else
            file_path = [file_path; cellstr([path filesep f.name])];
            file_name = [file_name; cellstr(f.name)];
        end
        
    end
end