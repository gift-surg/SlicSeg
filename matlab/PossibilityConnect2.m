% P0: possibility map
% lastSeg: segmentation of last frame(shape prior)

function P=PossibilityConnect2(P0,lastSeg)
[dis,maxdis]=SegShape2Distance(lastSeg);
isize=size(P0);
P=P0;
for i=1:isize(1)
    for j=1:isize(2)
        if(dis(i,j)==0)
            if(P(i,j)>0.5)
                P(i,j)=0.4*P(i,j);
            end
        else
            P(i,j)=P(i,j)+0.2*dis(i,j)/maxdis;
            if(P(i,j)>1.0)
                P(i,j)=1.0;
            end
        end
    end
end

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
