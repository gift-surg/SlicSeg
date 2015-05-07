function gmm = init_gmmBS(data,nMixtures);
%% Initialize a GMM using binary split
% 
% Input:
%  data - D x N array of D-dimensional features
%  nMixtures - the number of gaussian mixtures
% Output parameters:
%   gmm - A Gaussian mixture model structure.

import gsc.gmm.*

[D,N]=size(data);
[cluster, colours] = nema_vector_quantize( data, nMixtures );
for i = 1:nMixtures
  idx = find( cluster == i );
  gmm.mu( :, i ) = mean( data( :, idx ), 2 );
  if(length(idx)<=2), 
    gmm.sigma(:,:,i)=eye(D);
    gmm.pi(i)=0; %Effectively removing the gaussian
    fprintf('bs init:Gaussian number %d has been set to weight=0 (<=2 features assigned)\n',i);
  else
    gmm.sigma( :, :, i ) = cov( data( :, idx )' );
    if(max(diag(gmm.sigma(:,:,i)))<realmin),
      fprintf('bs init: Gaussian number %d was too unform, resetting its covariance\n',i);
      fprintf('This gaussian was assigned %d features\n',length(idx));
      %gmm.pi(i)=0;
      gmm.pi( i ) = length( idx );
      gmm.sigma(:,:,i)=0.0001*eye(D);  % I AM ASSUMING DATA BETWEEN [0,1]!!, QUICK HACK
      %fprintf('bs init: Gaussian number %d has been set to weight=0 (too uniform)\n',i);
      %fprintf('This gaussian was assigned %d features\n',length(idx));
      %gmm.pi(i)=0;
      %gmm.sigma(:,:,i)=eye(D);
    else
      gmm.sigma(:,:,i)=vag_covarFix(gmm.sigma(:,:,i));
      gmm.pi( i ) = length( idx );
    end
  end;
end
gmm.pi = gmm.pi / sum( gmm.pi );
