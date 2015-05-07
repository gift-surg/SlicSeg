function preProcess(obj,img)

if(~strcmp(obj.state,'init')),
  error('Can only preprocess from state init\n');
end

if(~strcmp(class(img),'double')),
  error('img should be of type double in preProcessing\n');
end

opts=obj.opts;
obj.img=img;
[h,w,nCh]=size(img);

if(strcmp(opts.spImg,'likelihoodImg')),
  switch(opts.posteriorMethod)
    case {'gmm_bs','gmm_bs_mixtured'}
      switch(opts.featureSpace)
        case 'rgb'
          obj.features=bj.extractPixels(img);
        case 'rgb_xy'
          obj.features=bj.extractPixels(img);
          obj.features=bj.vag_addSpatialFeatures(obj.features,img,10);
        otherwise
          error('Invalid feature space %s\n',opts.featureSpace);
      end
    case 'none'
      obj.features=[];
    otherwise
      error('Invalid posterior method %s\n',opts.posteriorMethod);
  end
elseif(strcmp(opts.spImg,'imgSmoothed'))
  sigma=opts.sigma;
  if(sigma>0)
    obj.smoothedImg=imfilter(img,fspecial('gaussian',round(2*sigma+1),sigma));
  else
    obj.smoothedImg=img;
  end
else
  error('Invalid image for computing shortest paths\n');
end

obj.state='pped';
