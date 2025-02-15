%Peter C. Doerschuk
%May 15, 2022

function [sj,sjp,sy,syp]=sphbes_vec(n,x)
%function [sj,sjp,sy,syp]=sphbes_vec(n,x)
%vectorized version of sphbes
%compute spherical bessel function J-Y, and their derivatives

if ~isscalar(n) || n<0
  error('sphbes_vec: n must be scalar >=0, n=%d.\n',n);
end

if any(x<0.0)
  error('sphbes_vec: x must be >=0, x=%g.\n',x);
end

ix1=find(x~=0.0);
x1=x(ix1);
ix0=find(x==0.0);

%Compute sj, sy, sjp, syp for x>0.
fact=sqrt(pi/2)./sqrt(x1);
order=n+0.5;
sj1=fact.*besselj(order,x1);
sy1=fact.*bessely(order,x1);
%by recursion formula
sjp1=(n./x1).*sj1-fact.*besselj(order+1,x1);
syp1=(n./x1).*sy1-fact.*bessely(order+1,x1);

%Compute sj, sy, sjp, syp for x=0.
if n==0
  sj0=1.0;
else
  sj0=0.0;
end
sjp0=0.0;
sy0=0.0;
syp0=0.0;

Nx=size(x,1);
sj=zeros(Nx,1);
sjp=zeros(Nx,1);
sy=zeros(Nx,1);
syp=zeros(Nx,1);

sj(ix0)=sj0;
sjp(ix0)=sjp0;
sy(ix0)=sy0;
syp(ix0)=syp0;

sj(ix1)=sj1;
sjp(ix1)=sjp1;
sy(ix1)=sy1;
syp(ix1)=syp1;
