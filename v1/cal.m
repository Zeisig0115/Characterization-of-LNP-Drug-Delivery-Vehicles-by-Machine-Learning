function [q, Iq] = cal(filename, nx)
    % rot_modified: 计算给定密度文件对应的散射强度 I(q)
    %    文件中存储的 rhoS 为 full size 数据（例如 100³），
    %    其中有效（非零）区域位于中心，其尺寸 M 通过文件名中的 "d" 后面的数字指定。
    %    采用固定比例将有效区域嵌入到大立方体中（大立方体尺寸 = nx³）。
    %
    % 输入:
    %   - filename: 包含密度数据的 .mat 文件路径，文件中应包含变量 rhoS
    %   - nx: 大立方体尺寸
    %
    % 输出:
    %   - q: 散射矢量大小数组
    %   - Iq: 对应于每个 q 的散射强度 I(q)
    %
    % 如果不对数据做窗函数处理，FFT 会受到边界不连续性的影响而产生谱泄露和失真，
    % 因此这里添加窗函数（例如 3D Hann 窗）对数据进行加窗处理。

    %% 初始化 FFT 相关参数
    nxf = nx;  % 这里 nxf 与 nx 保持一致
    nxd2 = nx / 2;
    nxfd2 = nxf / 2;
    iqcent = nxd2 + 1;

    %% 加载密度数据及确定有效模型尺寸 M
    data = load(filename);
    small_cube = data.rhoS;
    M = size(small_cube, 1);  % full size，例如 100

    %% 对小立方体数据加窗（使用 1D Hann 窗生成 3D 窗函数）
    % 如果 MATLAB 版本支持 hann() 函数，可以直接调用，否则可以手动定义：
    % hann1 = 0.5*(1-cos(2*pi*(0:M-1)/(M-1)))';
    hann1 = hann(M);  % 使用 MATLAB 内置的 hann 窗函数（生成列向量）
    % 生成 3D 窗函数（各方向乘积）
    [W1, W2, W3] = ndgrid(hann1, hann1, hann1);
    window3D = W1 .* W2 .* W3;
    % 对 small_cube 加窗
    small_cube = small_cube .* window3D;
    
    %% 将有效模型嵌入到大小为 nx³ 的大立方体中（保证模型位于立方体中心）
    rhoS = zeros(nx, nx, nx);
    center_position = nx / 2;
    start_pos = center_position - (M / 2) + 1;
    end_pos = center_position + (M / 2);
    rhoS(start_pos:end_pos, start_pos:end_pos, start_pos:end_pos) = small_cube;
    
    %% FFT 及散射强度计算
    % 计算三维 FFT 并取幅值平方得到 I(q) 的 3D 分布
    Iq3D = abs(fftn(rhoS)).^2;
    Iq3D = fftshift(Iq3D);  % 将零频移到中心位置

    % 计算用于校正体素效应的 sinc² 因子
    iq1 = (1:nx) - iqcent;
    q1ad2 = pi * iq1 / nx + 1e-8;  % 避免除零
    sincsqr = (sin(q1ad2) ./ q1ad2).^2;

    %% 归类计算 I(q)（球均值化）
    Iscatt = zeros(nxf, 1);
    iqmin = iqcent - nxfd2;
    iqmax = iqcent + nxfd2 - 1;
    symweight = 2 * ones(nx, 1);
    symweight(iqcent) = 1;  % 中心层权重设为1

    % 采用简单的线性插值方法，将 FFT 数据归类到对应的 q-bin 中
    for iq1 = iqmin:iqmax
        for iq2 = iqmin:iqmax
            qsqr12 = (iq1 - iqcent)^2 + (iq2 - iqcent)^2;
            for iq3 = iqcent:iqmax
                qabs = sqrt(qsqr12 + (iq3 - iqcent)^2) + 1;  % 加1防止 q=0
                iqabs = round(qabs);
                ishar = qabs - iqabs;
                share = abs(ishar);
                isignshare = sign(ishar);

                addI = Iq3D(iq1, iq2, iq3) * sincsqr(iq1) * sincsqr(iq2) * sincsqr(iq3) * symweight(iq3);
                
                % 检查边界后，将强度分配到对应的 q-bin
                if (iqabs >= 1 && iqabs <= nxf)
                    Iscatt(iqabs) = Iscatt(iqabs) + addI * (1 - share);
                end
                if (iqabs + isignshare >= 1 && iqabs + isignshare <= nxf)
                    Iscatt(iqabs + isignshare) = Iscatt(iqabs + isignshare) + addI * share;
                end
            end
        end
    end

    % 对归类后的强度做归一化（用 (iq-1)^2 作为归一化因子）
    Iq = zeros(nxfd2 - 1, 1);
    for iq = 2:nxfd2
        Iq(iq - 1) = Iscatt(iq) / (iq - 1)^2;
    end

    %% 计算 q 轴并绘图
    a = 1;  % 每个点对应的实际长度（例如1 Å/pt）
    dq = 1 / (nx * a);
    q = dq * (2:nxfd2);

    % 构造 legend 字符串，显示文件名
    legendStr = sprintf('%s', filename);

    % 绘制 I(q) 曲线（散射强度除以 4*pi 进行归一化显示）
    loglog(q, Iq / (4 * pi), 'DisplayName', legendStr);
    xlabel('q (Å^{-1})');
    ylabel('I(q) (a.u.)');
    hold on;
    legend('show');

    % 清除较大的中间变量（可选）
    clear Iq3D;
end
