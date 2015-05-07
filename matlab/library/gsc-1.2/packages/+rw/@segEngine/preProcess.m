function preProcess(obj,img)

if(~strcmp(obj.state,'init')),
  error('Can only preprocess from state init\n');
end

if(~strcmp(class(img),'double')),
  error('img should be of type double in preProcessing\n');
end

opts=obj.opts;
obj.img=img;
obj.state='pped';
