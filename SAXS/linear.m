function [q, Iq] = linear(filename, ratio)
    % rot: 计算给定密度文件对应的散射强度 I(q)
    %      文件中存储的 rhoS 为 full size 数据（例如 100³），
    %      其中有效（非零）区域位于中心，其尺寸 M 通过文件名中的 "d" 后面的数字指定。
    %      采用固定比例 ratio 将有效区域嵌入到大立方体中（大立方体尺寸 = ratio * M）。
    %
    % 输入:
    %   - filename: 包含密度数据的 .mat 文件路径，文件中应包含变量 rhoS
    %   - ratio: 填充立方体尺寸与模型非零区域尺寸 M 的比例因子（例如，6 表示大立方体尺寸 = 6 * M）
    %
    % 输出:
    %   - q: 散射矢量大小数组
    %   - Iq: 对应于每个 q 的散射强度 I(q)
    %
    % 若未提供 ratio 参数，则默认使用 6.

    if nargin < 2
        ratio = 6;
    end

    %% 加载密度数据及确定有效模型尺寸 M
    data = load(filename);
    small_cube = data.rhoS;
    M_full = size(small_cube, 1);  % full size，例如 100
    
    % 尝试从 filename 中提取 'd' 后面的数字作为实际非零区域尺寸 M
    token = regexp(filename, 'd(\d+)', 'tokens');
    if ~isempty(token)
        M = str2double(token{1}{1});
    else
        M = M_full;
    end
    fprintf('M = %d\n', M);
    
    %% 从 full size 数据中提取中心 M×M×M 的有效区域
    start_small = floor((M_full - M) / 2) + 1;
    end_small = start_small + M - 1;
    effective_model = small_cube(start_small:end_small, ...
                                 start_small:end_small, ...
                                 start_small:end_small);
    
    %% 根据有效区域尺寸 M 计算填充大立方体的尺寸
    nx = ratio * M;   % 如：M = 100 时 nx = 600；M = 20 时 nx = 120

    fprintf('Padding Cube Size = %d\n', nx);
    
    %% 初始化 FFT 相关参数
    nxf = nx;  
    nxd2 = nx / 2;
    nxfd2 = nxf / 2;
    iqcent = nxd2 + 1;
    
    % 将有效模型嵌入到大小为 nx³ 的大立方体中（保证模型位于立方体中心）
    rhoS = zeros(nx, nx, nx);
    center_position = nx / 2;
    start_pos = floor(center_position - M/2) + 1;
    end_pos = start_pos + M - 1;
    rhoS(start_pos:end_pos, start_pos:end_pos, start_pos:end_pos) = effective_model;
    
    %% FFT 及散射强度计算
    Iq3D = abs(fftn(rhoS)).^2;
    Iq3D = fftshift(Iq3D);  % 将零频移到中心位置

    % 计算用于校正体素效应的 sinc² 因子
    iq1 = (1:nx) - iqcent;
    q1ad2 = pi * iq1 / nx + 1e-8;  % 避免除零
    sincsqr = (sin(q1ad2) ./ q1ad2).^2;

    %% 归类计算 I(q)（球均值化）
    Iscatt = zeros(nxf, 1);

    % -- 新增: 统计每个 bin 接收的加权“体素数” --
    IscattCount = zeros(nxf, 1);  % 用于记录插值权重之和

    iqmin = iqcent - nxfd2;
    iqmax = iqcent + nxfd2 - 1;
    symweight = 2 * ones(nx, 1);
    symweight(iqcent) = 1;  % 中心层权重设为1

    for iq1 = iqmin:iqmax
        for iq2 = iqmin:iqmax
            qsqr12 = (iq1 - iqcent)^2 + (iq2 - iqcent)^2;
            for iq3 = iqcent:iqmax
                qabs = sqrt(qsqr12 + (iq3 - iqcent)^2) + 1;  % +1防止 q=0
                iqabs = round(qabs);
                ishar = qabs - iqabs; 
                share = abs(ishar);
                isignshare = sign(ishar);

                addI = Iq3D(iq1, iq2, iq3) * ...
                       sincsqr(iq1) * sincsqr(iq2) * sincsqr(iq3) * ...
                       symweight(iq3);
                
                % 主bin (iqabs)
                if (iqabs >= 1 && iqabs <= nxf)
                    Iscatt(iqabs) = Iscatt(iqabs) + addI * (1 - share);
                    % -- 新增: 记录插值权重(1 - share) --
                    IscattCount(iqabs) = IscattCount(iqabs) + (1 - share);
                end
                % 邻近bin (iqabs +/- 1), 由 isignshare 决定
                adjBin = iqabs + isignshare;
                if (adjBin >= 1 && adjBin <= nxf)
                    Iscatt(adjBin) = Iscatt(adjBin) + addI * share;
                    % -- 新增: 记录插值权重 share --
                    IscattCount(adjBin) = IscattCount(adjBin) + share;
                end
            end
        end
    end

    % -- 在归一化前可先看低 q 区的统计 (例如前10个bin) --
    numBinsToCheck = nx;
    fprintf('\n--- Low-q region bin stats (first %d bins) ---\n', numBinsToCheck);
    fprintf('binIdx   binCount   totalIntensity   avgIntensity\n');
    for i = 1:numBinsToCheck
        c = IscattCount(i);
        I_total = Iscatt(i);
        if c > 0
            I_avg = I_total / c;
        else
            I_avg = 0;
        end
        fprintf('%5d   %8.3f      %12.5g      %12.5g\n', i, c, I_total, I_avg);
    end

    %% --- 在打印完 bin 统计后，画出柱状图并叠加纯二次函数 ---
    numBinsPlot = nx;            % 要画的 bin 数
    binIdx     = (1:numBinsPlot)';      % bin 索引
    counts     = IscattCount(1:numBinsPlot);

    figure;
    bar(binIdx, counts, 'FaceColor', [0.2 0.6 1], 'EdgeColor', 'none');
    hold on;

    % -- 直接定义一个二次函数 y = a * x^2 --
    a = 6.20;   % 这里的 a 可以根据你想要的参考曲线 “高度” 自行调整
    y_quad = a * (binIdx.^2);
    plot(binIdx, y_quad, 'r-', 'LineWidth', 2);

    xlabel('bin index');
    ylabel('Effective binCount');
    title('True Effective Bin Counts');
    legend('BinCount', 'f(x)=2\pi x^2', 'Interpreter', 'tex', 'Location', 'NorthWest');
    xlim([0, numBinsPlot]);
    ylim([0, max(counts)*1.05]);

    
    %% 归类后的强度做归一化（用 (iq-1)^2 作为归一化因子）
    Iq = zeros(nxfd2 - 1, 1);
    for iq = 2:nxfd2
        Iq(iq - 1) = Iscatt(iq) / (iq - 1)^2;
    end

    %% 计算 q 轴并绘图
    a = 1;  % 每个点对应的实际长度（例如1 Å/pt）
    dq = 1 / (nx * a);
    q = dq * (2:nxfd2);
    
    % legendStr = sprintf('Original Algorithm');
    % loglog(q, Iq / (4 * pi), 'LineStyle', '-.','Color', [0.5 0.5 0.2], 'LineWidth', 1.5, 'DisplayName', legendStr);
    % xlabel('q (nm^{-1})');
    % ylabel('I(q) (a.u.)');
    % hold on;
    % legend('show');
    % clear Iq3D;
end
