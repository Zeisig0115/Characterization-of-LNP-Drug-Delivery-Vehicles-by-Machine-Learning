function [q, Iq] = improved(filename, ratio)
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

    %% 5. 利用向量化做球均值化 (径向归类)
    % 构造三维网格（原点在中心）
    [X, Y, Z] = ndgrid(-floor(nx/2):(ceil(nx/2)-1));
    R = sqrt(X.^2 + Y.^2 + Z.^2);  % 防止 R=0

    % 线性插值分配到相邻两个 bin
    rFloor = floor(R);
    rFrac  = R - rFloor;
    rFloor(rFloor < 1) = 1;
    rFloor(rFloor > nx) = nx;
    rFloorP1 = rFloor + 1;
    rFloorP1(rFloorP1 < 1) = 1;
    rFloorP1(rFloorP1 > nx) = nx;

    % 对应 fft 数据中每个体素的值，结合 sinc² 校正因子
    idxX = X + iqcent;
    idxY = Y + iqcent;
    idxZ = Z + iqcent;
    vals_Iq3D = Iq3D(sub2ind(size(Iq3D), idxX, idxY, idxZ));
    vals_sinc = sincsqr_1d(idxX) .* sincsqr_1d(idxY) .* sincsqr_1d(idxZ);
    vals = vals_Iq3D .* vals_sinc;

    % 分配权重
    wFloor   = 1 - rFrac;
    wFloorP1 = rFrac;

    % 用 accumarray 累加各 bin 的散射强度贡献
    Iscatt_part1 = accumarray(rFloor(:),   vals(:) .* wFloor(:),   [nx,1]);
    Iscatt_part2 = accumarray(rFloorP1(:), vals(:) .* wFloorP1(:), [nx,1]);
    Iscatt = Iscatt_part1 + Iscatt_part2;

    % 同时统计每个 bin 的有效体素数（插值权重之和）
    count_part1 = accumarray(rFloor(:),   wFloor(:),   [nx,1]);
    count_part2 = accumarray(rFloorP1(:), wFloorP1(:), [nx,1]);
    binCount = count_part1 + count_part2;

    % 打印低 q 区前 10 个 bin 的统计信息
    numBinsToCheck = nx;
    for i = 1:numBinsToCheck
        if binCount(i) > 0
            I_avg = Iscatt(i) / binCount(i);
        else
            I_avg = 0;
        end
        fprintf('%5d   %8.3f      %12.5g      %12.5g\n', i, binCount(i), Iscatt(i), I_avg);
    end

    %% 6. 归一化：直接用每个 bin 的有效体素数归一化
    nxd2 = nx / 2;
    IqVal = zeros(nxd2 - 1, 1);
    for iq = 2:nxd2
        if binCount(iq) > 0
            IqVal(iq - 1) = Iscatt(iq) / binCount(iq);
        else
            IqVal(iq - 1) = 0;
        end
    end

    %% 7. 计算 q 轴并绘图
    a = 1;               % 每个点对应的实际长度（例如 1 Å/pt）
    dq = 1 / (nx * a);   % q 的步长
    q = dq * (2 : nxd2); % q 轴

    Iq = IqVal;
    
    % 绘图
    loglog(q, Iq, 'LineWidth', 1.5, 'Color', 'b', 'DisplayName', 'Improved Algorithm');
    xlabel('q (nm^{-1})');
    ylabel('I(q) (a.u.)');
    legend('show');
    hold on;
end
