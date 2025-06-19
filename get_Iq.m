function [q, Iq] = get_Iq(filename, ratio)
    % cubic_effective_norm: Computes the scattering intensity I(q) 
    % and normalizes it using the effective voxel count in each bin.
    %
    % Inputs:
    %   - filename: Path to the .mat file containing density data 
    %               (should include the variable rhoS)
    %   - ratio: Ratio between the padded cube size and the non-zero 
    %            region size M (default = 6)
    %
    % Outputs:
    %   - q:   Array of scattering vector magnitudes
    %   - Iq:  Corresponding scattering intensities I(q)

    if nargin < 2
        ratio = 6;
    end

    %% 1. Load data and extract the effective region
    data = load(filename);
    small_cube = data.rhoS;
    M_full = size(small_cube, 1);
    
    % Extract the actual effective region size M from the filename 
    % (e.g., 'd20' implies M = 20)
    token = regexp(filename, 'd(\d+)', 'tokens');
    if ~isempty(token)
        M = str2double(token{1}{1});
    else
        M = M_full;
    end
    fprintf('M = %d\n', M);
    
    % Extract the center M×M×M region from the full-size data
    start_small = floor((M_full - M) / 2) + 1;
    end_small = start_small + M - 1;
    effective_model = small_cube(start_small:end_small, ...
                                 start_small:end_small, ...
                                 start_small:end_small);

    %% 2. Embed the effective model into a larger cube
    nx = ratio * M;
    fprintf('Padding Cube Size = %d\n', nx);
    
    rhoS = zeros(nx, nx, nx);
    center_position = nx / 2;
    start_pos = floor(center_position - M / 2) + 1;
    end_pos   = start_pos + M - 1;
    rhoS(start_pos:end_pos, start_pos:end_pos, start_pos:end_pos) = effective_model;

    %% 3. Perform FFT and compute 3D scattering amplitude squared (I(q) in 3D)
    Iq3D = abs(fftn(rhoS)).^2;
    Iq3D = fftshift(Iq3D);

    %% 4. Prepare sinc² correction factor
    iqcent = nx/2 + 1;
    iq1 = (1:nx) - iqcent;
    q1ad2 = pi * iq1 / nx + 1e-8;  % avoid division by zero
    sincsqr_1d = (sin(q1ad2) ./ q1ad2).^2;

    %% 5. Perform spherical averaging using vectorization
    % Construct a 3D meshgrid (origin at center)
    [X, Y, Z] = ndgrid(-floor(nx/2):(ceil(nx/2)-1));
    R = sqrt(X.^2 + Y.^2 + Z.^2);  % avoid R=0

    % Linearly interpolate between two adjacent bins
    rFloor = floor(R);
    rFrac  = R - rFloor;
    rFloor(rFloor < 1) = 1;
    rFloor(rFloor > nx) = nx;
    rFloorP1 = rFloor + 1;
    rFloorP1(rFloorP1 < 1) = 1;
    rFloorP1(rFloorP1 > nx) = nx;

    % Lookup FFT values for each voxel and apply sinc² correction
    idxX = X + iqcent;
    idxY = Y + iqcent;
    idxZ = Z + iqcent;
    vals_Iq3D = Iq3D(sub2ind(size(Iq3D), idxX, idxY, idxZ));
    vals_sinc = sincsqr_1d(idxX) .* sincsqr_1d(idxY) .* sincsqr_1d(idxZ);
    vals = vals_Iq3D .* vals_sinc;

    % Assign weights
    wFloor   = 1 - rFrac;
    wFloorP1 = rFrac;

    % Use accumarray to accumulate scattering intensity contributions per bin
    Iscatt_part1 = accumarray(rFloor(:),   vals(:) .* wFloor(:),   [nx,1]);
    Iscatt_part2 = accumarray(rFloorP1(:), vals(:) .* wFloorP1(:), [nx,1]);
    Iscatt = Iscatt_part1 + Iscatt_part2;

    % Also accumulate effective voxel counts (sum of interpolation weights)
    count_part1 = accumarray(rFloor(:),   wFloor(:),   [nx,1]);
    count_part2 = accumarray(rFloorP1(:), wFloorP1(:), [nx,1]);
    binCount = count_part1 + count_part2;

    %% 6. Normalize using effective voxel count in each bin
    nxd2 = nx / 2;
    IqVal = zeros(nxd2 - 1, 1);
    for iq = 2:nxd2
        if binCount(iq) > 0
            IqVal(iq - 1) = Iscatt(iq) / binCount(iq);
        else
            IqVal(iq - 1) = 0;
        end
    end

    %% 7. Compute q-axis and plot (optional)
    a = 1;               % Real length per voxel (e.g., 1 Å/pt)
    dq = 1 / (nx * a);   % q-step
    q = dq * (2 : nxd2); % q-axis

    Iq = IqVal;
end
