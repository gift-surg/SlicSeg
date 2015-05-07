function ok=start(obj,labelImg)

ok=false;
[h,w]=size(labelImg);
userDefineSeeds=0;
if(userDefineSeeds==1)
    inputLabelImg=imread('/Users/guotaiwang/Documents/MATLAB/ImageSeg/image_15/compare/31_seeds.png');
    [h1,w1]=size(inputLabelImg);
    hratio=double(h)/h1;
    wratio=double(w)/w1;
    for i=1:h
        for j=1:w

            i1=round(i/hratio);
            j1=round(j/wratio);
            if(inputLabelImg(i1,j1)==127)
                labelImg(i,j)=1;
            elseif(inputLabelImg(i1,j1)==255)
                labelImg(i,j)=2;
            else
                labelImg(i,j)=0;
            end   
        end
    end
end

geoImg=getGeoImg(obj,labelImg);
foreLabel=1;
backLabel=2;
stPointsFG=find(labelImg==foreLabel);
pts=zeros(2,length(stPointsFG));
if(~isempty(stPointsFG))
  [pts(1,:),pts(2,:)]=ind2sub([h w],stPointsFG);
end

dFG=computeDistances(geoImg,pts);

stPointsBG=find(labelImg==backLabel);
pts=zeros(2,length(stPointsBG));

if(~isempty(stPointsBG))
[pts(1,:),pts(2,:)]=ind2sub([h w],stPointsBG);
end

dBG=computeDistances(geoImg,pts);

seg=dFG<dBG;
obj.seg=255*uint8(seg);
if(userDefineSeeds==1)
    imwrite(obj.seg,'/Users/guotaiwang/Documents/MATLAB/ImageSeg/image_15/compare/31_seggeos.png');
end
ok=true;

function geoImg=getGeoImg(obj,labelImg)

opts=obj.opts;
switch(opts.spImg)
  case 'imgSmoothed'
      disp('imgSmoothed');
    geoImg=obj.smoothedImg;
  case 'likelihoodImg'
      disp('likelihoodImg');
    geoImg=sp.getPosteriorImage(obj.features,labelImg,opts); 
    obj.posteriorImage=geoImg;
end


function D=computeDistances(W,pts)
nb_iter_max =  1.2*max(size(W))^3;
[D,S,Q,stPoints] = sp.cpp.perform_front_propagation_2d_color(W,pts-1,[],nb_iter_max, [], []);
Q = Q+1;
stPoints=stPoints+1;

