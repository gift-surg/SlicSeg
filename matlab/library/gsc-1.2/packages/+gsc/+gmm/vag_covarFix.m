function sigma=vag_covarFix(sigma)
% Function to fix the covariance if it is getting singular
D=size(sigma,1);
fixrate=0.01;

while(rcond(sigma)<1e-8)
  m=max(diag(sigma))*fixrate;
  if(m<realmin), error('covarFix, max(diag)<realmin, dont know what to do\n'); end;
  sigma=sigma+m*eye(D);
  fprintf('Fixing gaussian covariance matrix, was close to singular\n');
end;
