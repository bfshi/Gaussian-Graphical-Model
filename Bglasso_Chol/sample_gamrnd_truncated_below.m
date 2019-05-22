function a = sample_gamrnd_truncated_below(a,b,lower)

[lo] = gamcdf([lower],a,b);

u = (1-lo)*rand(1)+lo;

u = min(u,1-1e-10);

a = gaminv(u,a,b);