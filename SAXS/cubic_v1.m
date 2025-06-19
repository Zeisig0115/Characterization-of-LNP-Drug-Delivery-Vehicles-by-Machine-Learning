function [q, Iq] = cubic_v1(filename, ratio, refineFactor, interpMethod)
    % cubic_v1_modified: 计算散射强度 I(q) 并利用每个 bin 的有效体素数归一化，
    % 并在最后一步对 q 轴和 I(q) 进行插值。插值方法可以选择更高阶的多项式插值。
    %
    % 输入:
    %   - filename: 包含密度数据的 .mat 文件路径，文件中应包含变量 rhoS
    %   - ratio: 填充立方体尺寸与模型非零区域尺寸 M 的比例因子（默认=6）
    %   - refineFactor: I(q) 插值因子，使得输出 q 点数为原来的 refineFactor 倍（默认=20）
    %   - interpMethod: 插值方法，可选 'spline'（默认）或 'poly5' 表示使用 5 次多项式局部插值
    %
    % 输出:
    %   - q:   更密的散射矢量大小数组
    %   - Iq:  对应于每个 q 的散射强度 I(q)
    
    if nargin < 2
        ratio = 6;
    end
    if nargin < 3
        refineFactor = 20;
    end
    if nargin < 4
        interpMethod = 'poly5';  % 默认采用 cubic spline 插值
    end

    %% 1. 加载数据及提取有效区域
    data = load(filename);
    small_cube = data.rhoS;
    M_full = size(small_cube, 1);
    
    % 从文件名中提取实际有效区域尺寸 M (例如 'd20' 则 M = 20)
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

    %% 5. 利用向量化做球均值化（径向归类）
    % 构造三维网格（原点在中心）
    [X, Y, Z] = ndgrid(-floor(nx/2):(ceil(nx/2)-1));
    R = sqrt(X.^2 + Y.^2 + Z.^2);  % 计算每个点到中心的距离

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

    % 分配权重（线性插值）
    wFloor   = 1 - rFrac;
    wFloorP1 = rFrac;

    % 利用 accumarray 累加各 bin 的散射强度贡献
    Iscatt_part1 = accumarray(rFloor(:),   vals(:) .* wFloor(:),   [nx,1]);
    Iscatt_part2 = accumarray(rFloorP1(:), vals(:) .* wFloorP1(:), [nx,1]);
    Iscatt = Iscatt_part1 + Iscatt_part2;

    % 同时统计每个 bin 的有效体素数（插值权重之和）
    count_part1 = accumarray(rFloor(:),   wFloor(:),   [nx,1]);
    count_part2 = accumarray(rFloorP1(:), wFloorP1(:), [nx,1]);
    binCount = count_part1 + count_part2;

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

    %% 7. 计算 q 轴
    a = 1;               % 每个点对应的实际长度（例如 1 Å/pt）
    dq = 1 / (nx * a);   % q 的步长
    q_coarse = dq * (2 : nxd2); % 原始 q 轴

    %% 8. 对 I(q) 做插值，使得采样点更多
    q_fine = linspace(q_coarse(1), q_coarse(end), length(q_coarse)*refineFactor);
    switch lower(interpMethod)
        case 'spline'
            % 传统 cubic spline 插值
            Iq_fine = interp1(q_coarse, IqVal, q_fine, 'spline');
        case 'poly5'
            % 利用 5 次多项式做局部插值
            % 对于每个细分点，在原始数据中选取 6 个点进行多项式拟合
            Iq_fine = zeros(size(q_fine));
            n = length(q_coarse);
            for j = 1:length(q_fine)
                % 找到离当前 q_fine 最近的 coarse 数据点索引
                [~, idx] = min(abs(q_coarse - q_fine(j)));
                % 选取周围的 6 个点作为拟合窗口
                win_start = max(1, idx-2);
                win_end   = min(n, idx+3);
                % 若窗口不足6个点，则进行调整
                if (win_end - win_start + 1) < 6
                    if win_start == 1
                        win_end = min(n, win_start+5);
                    elseif win_end == n
                        win_start = max(1, win_end-5);
                    end
                end
                win_idx = win_start:win_end;
                % 进行 5 次多项式拟合
                p = polyfit(q_coarse(win_idx), IqVal(win_idx), 5);
                Iq_fine(j) = polyval(p, q_fine(j));
            end
        otherwise
            error('未知的插值方法，请选择 ''spline'' 或 ''poly5''.');
    end

    % 输出更密的 q 和 I(q)
    q = q_fine;
    Iq = Iq_fine;
    
    % 绘图
    loglog(q, Iq, 'DisplayName', sprintf('Effective Norm (refined x%d)', refineFactor));
    xlabel('q (nm^{-1})');
    ylabel('I(q) (a.u.)');
    legend('show');
    hold on;
end
