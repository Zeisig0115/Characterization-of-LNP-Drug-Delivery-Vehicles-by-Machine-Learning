function [q, Iq] = rot_modified(filename, nx)
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

    %% 初始化 FFT 相关参数
    nxf = nx;  % 这里 nxf 与 nx 保持一致
    nxd2 = nx / 2;
    nxfd2 = nxf / 2;
    iqcent = nxd2 + 1;

    %% 加载密度数据及确定有效模型尺寸 M
    data = load(filename);
    small_cube = data.rhoS;
    M = size(small_cube, 1);  % full size，例如 100
    
    % 将有效模型嵌入到大小为 nx³ 的大立方体中（保证模型位于立方体中心）
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

    % 构造 legend 字符串，显示文件名和 ratio 值
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
