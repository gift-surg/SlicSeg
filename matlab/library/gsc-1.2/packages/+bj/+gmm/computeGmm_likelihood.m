function likeli=computeGmm_likelihood(features,gmm)

import bj.gmm.*

likeli=zeros(1,size(features,2));
for i = 1:length( gmm.pi )
  likeli=likeli+vag_normPdf(features,gmm.mu(:,i),gmm.sigma(:,:,i),gmm.pi(i));
end
