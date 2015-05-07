function posteriors=compute_gmmPosteriors(features,fgGmm,bgGmm)
% Function to compute posteriors of each feature
% given the gmm
% Inputs:
%  features: a D x N array of features (D=dimensionality)
%  fgGmm, bgGmm: gmm structures for the foreground and background
%  gmm
import gscSeq.gmm.*;

fgLikeli=computeGmm_likelihood(features,fgGmm);
bgLikeli=computeGmm_likelihood(features,bgGmm);

divByZero=(fgLikeli==0 & bgLikeli==0);
fgLikeli(divByZero)=1;
bgLikeli(divByZero)=1;
posteriors=fgLikeli./(fgLikeli+bgLikeli);

%function likeli=computeGmm_likelihood(features,gmm)
%
%likeli=zeros(1,size(features,2));
%for i = 1:length( gmm.pi )
  %likeli=likeli+vag_normPdf(features,gmm.mu(:,i),gmm.sigma(:,:,i),gmm.pi(i));
%end
