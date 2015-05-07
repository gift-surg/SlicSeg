function [Wstar,starInfo]=updateStarEdges(labelImg,nbrHood,geoImg,geoGamma,prevSeg,starInfo)

% Function to compute the adjacency matrix for the 0-infty edges

[h w]=size(labelImg);
N=h*w;

switch(nbrHood)
  case 4
    roffset = [ 1,  0 ]; % 4 nbrhood
    coffset = [  0, 1  ];
  case 8
    roffset = [ 1, 1, 0, -1 ];
    coffset = [  0, 1, 1, 1 ];
end
[lEdges,rEdges,colorWeights,spWeights]=...
    gscSeq.cpp.mex_setupTransductionGraph(geoImg,int32(roffset'),int32(coffset'));
avgEucEdge_sqr=sum(roffset.*roffset+coffset.*coffset)/length(roffset); 
avgGeoEdge_sqr=mean(colorWeights);
rescale_geo=avgEucEdge_sqr/avgGeoEdge_sqr;

stPointsFG=find(labelImg==1);
numPts=length(stPointsFG);
pts=zeros(3,length(stPointsFG));

if(~isempty(stPointsFG))
  [pts(1,:),pts(2,:),pts(3,:)]=ind2sub([h w],stPointsFG);
else
  pts=zeros(3,0);
end

pts=pts(1:2,:);

[dFG,rootPoints,qFG]=gscSeq.shortestPaths_normalized(geoImg,pts,geoGamma,...
                     nbrHood,rescale_geo,starInfo.dFG,starInfo.qFG,prevSeg==255);

% qFG encodes the geodesic tree, rootPoints encodes the root node for each point
starInfo.dFG=dFG;
starInfo.qFG=qFG;
starInfo.rootPointsFG=rootPoints;

lInds=[1:N];
rInds=qFG(lInds);
mask=(lInds~=rInds);
lInds=lInds(mask);
rInds=rInds(mask);

clear stPointsFG pts dFG qFG;

pairs=[rInds' lInds'];
pairs=unique(pairs,'rows');

if(isempty(pairs)), pairs=zeros(0,2); end;

Wstar=sparse(pairs(:,1),pairs(:,2),logical(ones(size(pairs,1),1)),N,N);
