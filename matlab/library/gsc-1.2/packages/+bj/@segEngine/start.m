function ok=start(obj,labelImg)

ok=false;
[h,w]=size(labelImg);

stTime=clock;

opts=obj.opts;
posteriorImage=bj.getPosteriorImage(obj.features,labelImg,obj.opts);
obj.posteriorImage=posteriorImage;
posteriorImage=posteriorImage(:)';
%fprintf('Posteriors computed in %.2f seconds\n',etime(clock,stTime));

stTime=clock;

bgClamp=(labelImg(:)==2);
fgClamp=(labelImg(:)==1);

probDensities=[-log(1-posteriorImage); -log(posteriorImage)];
probDensities(probDensities>100)=100;

probDensities(2,bgClamp)=inf;probDensities(1,bgClamp)=0;
probDensities(1,fgClamp)=inf;probDensities(2,fgClamp)=0;

probDensities=int32(round(probDensities*opts.gcScale));

[E_n,E_w]=bj.convertMatrix_toEdgePairs(obj.W);
E_n=E_n';
E_w=E_w';

dgcHandle=bj.cpp.mexDGC('initialize',uint32(h*w),E_n); 
[cut,flow]=bj.cpp.mexDGC('minimizeDynamic',dgcHandle,uint32([1:h*w]), ...
             checkInfSTE(probDensities), E_n, E_w); 
seg = reshape( ~cut, h, w );
bj.cpp.mexDGC('cleanup',dgcHandle)

if(opts.postProcess)
  fgBrushMask=(labelImg==1);
  [fgL,fgNum]=bwlabel(fgBrushMask);
  segL=bwlabel(seg);
  seg(:)=0;
  for i=1:fgNum
    brIndex=find(fgL==i,1);
    segLabel=segL(brIndex);
    if(segLabel~=0),
      seg(segL==segLabel)=true;
    end
  end
end

obj.seg=255*uint8(seg);
fprintf('Graph cut computed in %.2f seconds\n',etime(clock,stTime));
ok=true;

function STE=checkInfSTE(STE)
% STE passed should be int32

steDiff=STE(1,:)-STE(2,:);
infInd=steDiff>(intmax-1);
STE(1,infInd)=intmax-1+STE(2,infInd);
infInd=steDiff<(-intmax+1);
STE(2,infInd)=intmax-1+STE(1,infInd);
