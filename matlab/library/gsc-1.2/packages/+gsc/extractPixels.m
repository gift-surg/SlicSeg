function pixels=extractPixels(C)
% Function to take a video h x w x dim x nFrames
% and output a matrix of the order
% dim x (h*w*nFrames) which contains
% a column major ordering of the video pixels

  [h,w,dim,nFrames]=size(C);

  pixels=zeros(dim,h*w*nFrames);
  
  for i=1:nFrames
    inds=[1:h*w];
    pixels(:,1+(i-1)*h*w:i*h*w)=gsc.nema_impixel(C(:,:,:,i),inds)';
  end
