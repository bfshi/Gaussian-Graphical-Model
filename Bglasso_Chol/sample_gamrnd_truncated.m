function a = sample_gamrnd_truncated(a,b,lower,upper)

l = gamcdf([lower,upper],a,b);

u = (l(2) -l(1))*rand(1)+l(1);
%u = min(max(1e-200,u),1-1e-10);

u = min(u,1-1e-10);

a = gaminv(u,a,b);

if a == Inf
   error('a == Inf');
end