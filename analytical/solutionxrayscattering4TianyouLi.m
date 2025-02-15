%Peter C. Doerschuk
%May 15, 2022

function solutionxrayscattering4TianyouLi()

%units: Angstroms and inverse Angstroms

R4plot=100;
x=0:0.5:2*R4plot;
k=0.004:1/(100*R4plot):20/R4plot;

%constant electron scattering at all radii
R=40;
rho0=1;

% rhoS=get_rhoS(x,rho0,R);
% figure;
% plot(x,rhoS);
% xlabel('x (Angstroms)');
% ylabel('\rho_S(x)');
% ylim([-.1 1.3]);
% fprintf(1,'about to pause\n');pause;
% print -dpdf solutionxrayscattering_realspace1.pdf

IS=get_IS(k,rho0,R);
loglog(k, IS, 'DisplayName', 'Analytical');
xlabel('k (inverse Angstroms)');
ylabel('I_S(k)');
hold on
end


function IS=get_IS(k,rho0,R)
IS=get_PS(k,rho0,R).^2;
end


function PS=get_PS(k,rho0,R)
kprime = 2*pi*k*R;
[sphericalbessel, ~, ~, ~]=sphbes_vec(1,kprime);
PS=(4*pi*rho0*R^3).*sphericalbessel./kprime;
end

