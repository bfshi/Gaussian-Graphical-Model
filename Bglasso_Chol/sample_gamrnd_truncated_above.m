function a = sample_gamrnd_truncated_above(a,b,upper)
%  x^(a-1) exp(-x/b)
[Up] = gamcdf([upper],a,b);

u = Up*rand(1);
%u = min(max(1e-200,u),1-1e-200);



a = gaminv(u,a,b);

if a == Inf
   error('a == Inf');
end