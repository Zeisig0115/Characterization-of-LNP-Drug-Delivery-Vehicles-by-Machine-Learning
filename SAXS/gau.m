function [q, Iq] = gau(filename, ratio)
    % cubic_effective_norm: 计算散射强度 I(q) 并利用每个 bin 的有效体素数归一化
    %
    % 输入:
    %   - filename: 包含密度数据的 .mat 文件路径，文件中应包含变量 rhoS
    %   - ratio: 填充立方体尺寸与模型非零区域尺寸 M 的比例因子（默认=6）
    %
    % 输出:
    %   - q:   散射矢量大小数组
    %   - Iq:  对应于每个 q 的散射强度 I(q)
    
    if nargin < 2
        ratio = 6;
    end

    %% 1. 加载数据及提取有效区域
    data = load(filename);
    small_cube = data.rhoS;
    M_full = size(small_cube, 1);
    
    % 从文件名中提取实际有效区域尺寸 M (如 'd20' 则 M = 20)
    token = regexp(filename, 'd(\d+)', 'tokens');
    if ~isempty(token)
        M = str2double(token{1}{1});
    else
        M = M_full;
    end
    fprintf('M = %d\n', M);
    
    % 从 full size 数据中提取中心 M×M×M 的有效区域
    start_small = floor((M_full - M) / 2) + 1;
    end_small = start_small + M - 1;
    effective_model = small_cube(start_small:end_small, ...
                                 start_small:end_small, ...
                                 start_small:end_small);

    %% 2. 将有效模型嵌入到更大的立方体中
    nx = ratio * M;
    fprintf('Padding Cube Size = %d\n', nx);
    
    rhoS = zeros(nx, nx, nx);
    center_position = nx / 2;
    start_pos = floor(center_position - M / 2) + 1;
    end_pos   = start_pos + M - 1;
    rhoS(start_pos:end_pos, start_pos:end_pos, start_pos:end_pos) = effective_model;

    %% 3. FFT 并计算 3D 散射幅度平方 (I(q) 的 3D 分布)
    Iq3D = abs(fftn(rhoS)).^2;
    Iq3D = fftshift(Iq3D);

    %% 4. 准备 sinc² 校正因子
    iqcent = nx/2 + 1;
    iq1 = (1:nx) - iqcent;
    q1ad2 = pi * iq1 / nx + 1e-8;  % 防止除零
    sincsqr_1d = (sin(q1ad2) ./ q1ad2).^2;

    %% 5. 利用向量化构造三维网格（原点在中心）并计算每个体素的修正值
    [X, Y, Z] = ndgrid(-floor(nx/2):(ceil(nx/2)-1));
    % 计算每个体素的径向距离（单位：格点数）
    R = sqrt(X.^2 + Y.^2 + Z.^2);
    
    % 对应 fft 数据中每个体素的值，结合 sinc² 校正因子
    idxX = X + iqcent;
    idxY = Y + iqcent;
    idxZ = Z + iqcent;
    vals_Iq3D = Iq3D(sub2ind(size(Iq3D), idxX, idxY, idxZ));
    vals_sinc = sincsqr_1d(idxX) .* sincsqr_1d(idxY) .* sincsqr_1d(idxZ);
    vals = vals_Iq3D .* vals_sinc;
    
    %% 6. 使用高斯分布的权重进行球均值化，同时采用 effective count 归一化
    % FFT 的 q 轴步长（这里单位依赖于 a，每个点对应实际长度 a）
    a = 1;  % 每个点对应的实际长度，例如 1 Å/pt
    dq = 1 / (nx * a);
    % 根据文中定义，Δq 可取 2π/(nx*a)
    Delta_q = 2 * pi / (nx * a);
    
    % 为获得更窄的高斯核，设置 narrowingFactor 较大（原来为 2，这里设为 4）
    narrowingFactor = 3;
    
    % 将每个体素的径向距离转换为 q 值
    q_voxel = dq * R;  % 单位为 nm^{-1}（若 a 为 nm/pt）
    
    % 预分配：只考虑 q 正半轴内的 bin（这里取前 floor(nx/2) 个）
    n_bins = floor(nx/2);
    Iscatt_gauss = zeros(n_bins, 1);
    binCount_gauss = zeros(n_bins, 1);
    
    % 对于每个目标 bin，定义其 q 值为 q_target = dq * bin，
    % 并对所有体素使用高斯权重分配
    for bin = 2:n_bins  % 从 2 开始，通常低 q 区不可靠
        q_target = dq * bin;
        % 使用更窄的高斯核：以 q_target 为均值，标准差约为 Delta_q/√(narrowingFactor/2)（此处用指数系数 narrowingFactor）
        % 注意：这里权重还除以 q_voxel^2 (几何补偿)，避免 q_voxel 为 0 时除零，加上 eps
        w = exp( -narrowingFactor * ((q_voxel - q_target) / Delta_q).^2 );
        % 累加加权散射强度
        Iscatt_gauss(bin) = sum(vals(:) .* w(:));
        % 有效计数（即权重之和）
        binCount_gauss(bin) = sum(w(:));
    end
    
    % 用 effective count 归一化：计算每个 bin 的平均散射强度
    IqVal = zeros(n_bins - 1, 1);
    for bin = 2:n_bins
        if binCount_gauss(bin) > 0
            IqVal(bin - 1) = Iscatt_gauss(bin) / binCount_gauss(bin);
        else
            IqVal(bin - 1) = 0;
        end
    end
    
    %% 7. 计算 q 轴并绘图
    % 注意：q 轴对应 bin 从 2 到 n_bins
    q = dq * (2:n_bins);
    Iq = IqVal;
    
    % 绘图
    loglog(q, Iq, 'DisplayName', 'Narrower Gaussian Weighted');
    xlabel('q (nm^{-1})');
    ylabel('I(q) (a.u.)');
    legend('show');
    hold on;
end
