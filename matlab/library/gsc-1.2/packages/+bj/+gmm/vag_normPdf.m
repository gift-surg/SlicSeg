function p = vag_normPdf( x, mu, sigma,weight )

% NEMA_LOGNORM A function that calculates the normal
% function 
%
%    P = vag_normPdf( X, MU, SIGMA,weight) calculates the 
%    normal function N-D dimensional columnwise points
%    in X according to
%
%      P(X) =  weight* 1 / sqrt( (2*pi)^N * det(SIGMA) ) ) -
%             0.5 * (X-MU)' * SIGMA^(-1) * (X-MU)
%
%    MU is a Dx1 mean column vector and SIGMA is the DxD covariance
%    matrix.



% Author: Varun Gulshan <varun@robots.ox.ac.uk>
% Date: 03 Jun 05

import bj.gmm.*;

d = length( mu );

if size( x, 1 ) ~= d | size( sigma, 1 ) ~= size( sigma, 2 ) ...
      | size( sigma, 1 ) ~= d
  error( 'Inconsistent input dimensions' );
end

n = size(mu,1);
npoints = size(x,2);

if(weight<eps)
  p = zeros( 1, npoints );
else
  if(det(sigma)>realmin)
    t1 = ( (2*pi)^n * abs(det(sigma)) ).^(-0.5);
    t2 = inv(sigma);
    %p = zeros( 1, npoints );
    p=vag_norm_fast( x, mu, t2, t1 );
  else
    error('varun_normPdf: bad sigma passed to me, det(sigma)<realmin\n');
  end
end
