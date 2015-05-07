function [D,stPoints,Q]=shortestPaths_normalized(W,start_points,geoGamma,nbrHoodSize,rescale_geo,D,Q,prevSeg)

end_points = [];

nb_iter_max = min(inf, 1.2*max(size(W))^3);

if(nargin<=5),
  [h w nCh]=size(W);
  D=inf*ones([h w]);
  prevSeg=logical(zeros([h w]));
  Q=0*ones([h w]);
end

[D,S,Q,stPoints] = gscSeq.cpp.mex_spNormalized_constrained(W,start_points-1,end_points-1,nb_iter_max,geoGamma,nbrHoodSize,rescale_geo,D,Q-1,prevSeg);
Q = Q+1;
stPoints=stPoints+1;
