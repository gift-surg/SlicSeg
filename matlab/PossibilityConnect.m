% P0: possibility map
% Label: foreground and background strokes

function P=PossibilityConnect(I,P0,Label)
PL=P0>=0.5;
pSe= strel('disk',3);
pMask=imclose(PL,pSe);
isize=size(P0);
L=uint8(zeros(isize));
fg=I(find(Label>0));
fg_mean=mean(fg);
fg_std=sqrt(var(double(fg)));
fg_min=fg_mean-fg_std*3;
fg_max=fg_mean+fg_std*2;
seeds=0;
indexH=uint16(zeros(isize(1)*isize(2),1));
indexW=uint16(zeros(isize(1)*isize(2),1));
P=P0;
for i=1:isize(1)
    for j=1:isize(2)
        if(Label(i,j)>0)
            seeds=seeds+1;
            L(i,j)=1;
            P(i,j)=1.0;
            indexH(seeds,1)=i;
            indexW(seeds,1)=j;
        end
    end
end
current=1;

while(current<=seeds)
    i=indexH(current,1);
    j=indexW(current,1);
    if(i+1<isize(1) && L(i+1,j)==0 && pMask(i+1,j)>0 && I(i+1,j)>fg_min && I(i+1,j)<fg_max)
        L(i+1,j)=1;
        seeds=seeds+1;
        indexH(seeds,1)=i+1;
        indexW(seeds,1)=j;  

    end
    if(i-1>0 && L(i-1,j)==0 && pMask(i-1,j)>0 && I(i-1,j)>fg_min && I(i-1,j)<fg_max)
        L(i-1,j)=1;
        seeds=seeds+1;
        indexH(seeds,1)=i-1;
        indexW(seeds,1)=j;  
    end
    if(j+1<isize(2) && L(i,j+1)==0 && pMask(i,j+1)>0 && I(i,j+1)>fg_min && I(i,j+1)<fg_max)
        L(i,j+1)=1;
        seeds=seeds+1;
        indexH(seeds,1)=i;
        indexW(seeds,1)=j+1;  
    end
    if(j-1>0 && L(i,j-1)==0 && pMask(i,j-1)>0 && I(i,j-1)>fg_min && I(i,j-1)<fg_max)
        L(i,j-1)=1;
        seeds=seeds+1;
        indexH(seeds,1)=i;
        indexW(seeds,1)=j-1;  
    end
    if(i+1<isize(1) && j+1<isize(2) && L(i+1,j+1)==0 && pMask(i+1,j+1)>0 && I(i+1,j+1)>fg_min && I(i+1,j+1)<fg_max)
        L(i+1,j+1)=1;
        seeds=seeds+1;
        indexH(seeds,1)=i+1;
        indexW(seeds,1)=j+1;  
    end
    if(i-1>0 && j+1<isize(2) && L(i-1,j+1)==0 && pMask(i-1,j+1)>0 && I(i-1,j+1)>fg_min && I(i-1,j+1)<fg_max)
        L(i-1,j+1)=1;
        seeds=seeds+1;
        indexH(seeds,1)=i-1;
        indexW(seeds,1)=j+1;  
    end
    if(i+1<isize(1) && j-1>0 && L(i+1,j-1)==0 && pMask(i+1,j-1)>0 && I(i+1,j-1)>fg_min && I(i+1,j-1)<fg_max)
        L(i+1,j-1)=1;
        seeds=seeds+1;
        indexH(seeds,1)=i+1;
        indexW(seeds,1)=j-1;  
    end
    if(i-1>0 && j-1>0 && L(i-1,j-1)==0 && pMask(i-1,j-1)>0 && I(i-1,j-1)>fg_min && I(i-1,j-1)<fg_max)
        L(i-1,j-1)=1;
        seeds=seeds+1;
        indexH(seeds,1)=i-1;
        indexW(seeds,1)=j-1;  
    end
    current=current+1;
end

for i=1:isize(1)
    for j=1:isize(2)
        if(P(i,j)>0.5 && L(i,j)==0)
            P(i,j)=0.4*P(i,j);
        end
    end
end