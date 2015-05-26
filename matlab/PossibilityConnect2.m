% P0: possibility map
% lastSeg: segmentation of last frame(shape prior)

function P=PossibilityConnect2(P0,lastSeg)
[dis,maxdis]=SegShape2Distance(lastSeg);
isize=size(P0);
P=P0;
outsideIndex=intersect(find(dis==0),find(P>0.5));
P(outsideIndex)=0.4*P(outsideIndex);
insideIndex=intersect(find(dis>0) , find(P<0.8));
P(insideIndex)=P(insideIndex)+0.2*dis(insideIndex)/maxdis;

function [dis,maxdis]=SegShape2Distance(lastSeg)
Isize=size(lastSeg);
dis=zeros(Isize);
se= strel('disk',1);
temp0=lastSeg;
temp1=imerode(temp0,se);
currentdis=0;
while(~isempty(find(temp1>0)))
    dis0=temp0-temp1;
    currentdis=currentdis+1;
    dis(dis0>0)=currentdis;
    temp0=temp1;
    temp1=imerode(temp0,se);
end
maxdis=currentdis;
