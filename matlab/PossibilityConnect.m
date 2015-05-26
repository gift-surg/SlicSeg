% P0: possibility map
% Label: foreground and background strokes

function P=PossibilityConnect(I,P0,Label)
PL=P0>=0.5;
pSe= strel('disk',3);
pMask=imclose(PL,pSe);
[H,W]=size(P0);
HW=H*W;
indexHW=uint32(zeros(HW,1));
seedsIndex=find(Label>0);
seeds=length(seedsIndex);
indexHW(1:seeds)=seedsIndex(1:seeds);
L=uint8(zeros(H,W));
P=P0;
L(seedsIndex)=1;
P(seedsIndex)=1.0;

fg=I(seedsIndex);
fg_mean=mean(fg);
fg_std=sqrt(var(double(fg)));
fg_min=fg_mean-fg_std*3;
fg_max=fg_mean+fg_std*2;

current=1;
while(current<=seeds)
    currentIndex=indexHW(current);
    NeighbourIndex=[currentIndex-1,currentIndex+1,...
        currentIndex+H,currentIndex+H-1,currentIndex+H+1,...
        currentIndex-H,currentIndex-H-1,currentIndex-H+1];
    for i=1:8
        tempIndex=NeighbourIndex(i);
        if(tempIndex>0 && tempIndex<HW && L(tempIndex)==0 && pMask(tempIndex)>0 && I(tempIndex)>fg_min && I(tempIndex)<fg_max)
            L(tempIndex)=1;
            seeds=seeds+1;
            indexHW(seeds,1)=tempIndex;
        end
    end
    current=current+1;
end

Lindex=find(L==0);
P(Lindex)=P(Lindex)*0.4;