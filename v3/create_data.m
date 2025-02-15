clear;
clc;

% 1. 批量读取 ./type_1_model/*.mat 文件
fileList = dir('./type_4_model/*.mat');

% 2. 准备输出目录及输出 hdf5 文件路径
outputDir = './output';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
hdf5_filename = fullfile(outputDir, 'type4.h5');

% 若文件已存在，则删除，避免旧数据冲突
if exist(hdf5_filename, 'file')
    delete(hdf5_filename);
end

% 3. 逐个处理每个 .mat 文件
for k = 1:length(fileList)
    % 获取 .mat 文件完整路径
    filename = fullfile(fileList(k).folder, fileList(k).name);

    % 4. 调用 get_Iq 函数计算 q 和 Iq，这里 ratio 传 6（如有需要可修改）
    [q, Iq] = get_Iq(filename, 6);

    % 5. 将 q 和 Iq 拼成两列的数据 (N×2)，方便后续一次性读取
    data_qIq = [q(:), Iq(:)];

    % 6. 取文件名（不含路径和后缀），作为 dataset 名的一部分
    [~, name, ~] = fileparts(filename);
    dataset_name = ['/', name, '_qIq'];  % 如 /XX_qIq

    % 7. 在 HDF5 文件中创建相应的数据集，并写入
    h5create(hdf5_filename, dataset_name, size(data_qIq));
    h5write(hdf5_filename, dataset_name, data_qIq);

    disp(['Processed and saved: ', name]);
end
