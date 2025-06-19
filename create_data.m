clear;
clc;

% 1. Batch read all .mat files from ./type_i_model/
fileList = dir('./type_4_model/*.mat');

% 2. Prepare output directory and HDF5 file path
outputDir = './output';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
hdf5_filename = fullfile(outputDir, 'raw4.h5');

% If the file already exists, delete it to avoid conflict with old data
if exist(hdf5_filename, 'file')
    delete(hdf5_filename);
end

% 3. Process each .mat file one by one
for k = 1:length(fileList)
    % Get the full path of the .mat file
    filename = fullfile(fileList(k).folder, fileList(k).name);

    % 4. Call get_Iq function to compute q and Iq (use ratio = 6, modify if needed)
    [q, Iq] = get_Iq(filename, 6);

    % 5. Combine q and Iq into a two-column array (N×2) for easier access
    data_qIq = [q(:), Iq(:)];

    % 6. Extract filename (without path or extension) for dataset naming
    [~, name, ~] = fileparts(filename);
    dataset_name = ['/', name, '_qIq'];  % e.g., /XX_qIq

    % 7. Create dataset in HDF5 file and write the data
    h5create(hdf5_filename, dataset_name, size(data_qIq));
    h5write(hdf5_filename, dataset_name, data_qIq);

    disp(['Processed and saved: ', name]);
end
