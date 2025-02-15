clear;
clc;

nx = input('# of pts for the model in each dimension -->');
nxf = input('# of q points in I(q) -->');  
% can be smaller than nx, to speed up calculation
% (gives only small-q results)

nxd2 = nx/2;
nxfd2 = nxf/2;
iqcent = nxd2+1;

%----------------------------------------------------
% From outside: scattering density rho
% make sure nx is the same in the external file
%----------------------------------------------------

filename = "./type1_set/d581_1.mat";
data = load(filename);
small_cube = data.rhoS;
M = size(small_cube, 1); % small cube size = 100
rhoS = zeros(nx, nx, nx);
center_position = nx / 2;
start_pos = center_position - (M / 2) + 1;
end_pos = center_position + (M / 2);
rhoS(start_pos:end_pos, start_pos:end_pos, start_pos:end_pos) = small_cube;

% ----------------------------------------
% Calculate scattering amplitude A(q) (complex)
% by Fourier transformation of rho(x)
% I(q)=|A(q)|^2 
% ----------------------------------------
 
Iq3D = abs(fftn(rhoS)).^2;
Iq3D = fftshift(Iq3D);  %center of grid (nxd2+1) is at q=0
 
%-------------------------------------------------------
% Prepare convolution with elementary square of area a^2
% by multiplication with sinc function in each dimension
%-------------------------------------------------------
iq1=(1:nx)-iqcent;
q1ad2=pi*iq1/nx+0.00000001;  % q1*a/2 = iq1c*2pi/(nx*a) *a/2
sincsqr=(sin(q1ad2)./q1ad2).^2;
 
%----------------------------------------------------------
% Now calculate I(|q|)*q^2 by summing over all orientations
% this is I(q)*q^2, because the density of the cubic grid varies like q^2
%----------------------------------------------------------
Iscatt=zeros(nxf,1);
 
iqmin = iqcent-nxfd2;
iqmax = iqcent+nxfd2-1;
symweight = 2*ones(nx,1);  % weight from iqcent+Diq3 and iqcent-Diq3
symweight(iqcent) = 1;  % centerpoint weight; allows limiting iq3 to half the range
 
for iq1=iqmin:iqmax 
    for iq2=iqmin:iqmax 
    qsqr12 = (iq1 - iqcent)^2+(iq2 - iqcent)^2; 
        for iq3=iqcent:iqmax  
 
        % Channel sharing:
        % When qabs falls between two integer values, distribute intensity between them
 
        qabs = sqrt(qsqr12 + (iq3-iqcent)^2)+1; %|q|   +1: prevent Iscatt(iqabs=0)
        iqabs = round(qabs);
        ishar = qabs - iqabs;  %
        share = abs(ishar);  % how much to share
        isignshare = sign(ishar);  % +1 if q > iq; 0 if exactly on
 
        addI = Iq3D(iq1,iq2,iq3)*sincsqr(iq1)*sincsqr(iq2)*sincsqr(iq3)*symweight(iq3);  %sincsqr: effect of elementary square
 
        Iscatt(iqabs) = Iscatt(iqabs)+addI*(1-share);
        Iscatt(iqabs+isignshare) = Iscatt(iqabs+isignshare)+addI*share; % 0 if exactly on
 
        end
    end
end
 
% ----------------------------------------
% calculate I(q)
% ----------------------------------------
for iq=2:nxfd2
 Iq(iq-1) = Iscatt(iq)/(iq-1)^2;  %iq=0 really q=0
end


clear Iq3

% how many Angstrom / pts. e.g. 10 Ang/pt
a = 1;  

% Calculate dq
dq = 1 / (nx * a);

q = dq * (2:nxfd2);

figure
loglog(q, Iq/(4*pi), 'DisplayName', 'Reference');
xlabel('q (Å^{-1})');
ylabel('I(q) (a.u.)');
hold on
xlim
legend;


% solutionxrayscattering4TianyouLi();

