% Convert HS dataset to patches

% List all '.mat' file in folder
file_folder=fullfile('H:\Program Files\MATLAB\R2015b\bin\Hyperspec_Chikusei_MATLAB\Chikusei_MATLAB\train');
file_list=dir(fullfile(file_folder,'*.mat'));
file_names={file_list.name};

% store cropped images in folders
for i = 1:1:numel(file_names)
    name = file_names{i};
    name = name(1:end-4);
    s=strcat('mkdir patches_',name);
    system(s);
    load(strcat('./train/',file_names{i}));
    crop_image(img, 64, 32, 0.25, name);
end
