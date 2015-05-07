function runOffline()
% Demonstrates how to call the segmentation without the user interface
% You need an image, and an annotation file with the brush strokes
% The annotation file should be a single channel image with the following
% values:
% 0 -> for pixels whose labeling is unknown, i.e they need to be segmented
% 1 -> pixels contrained to be foreground, these are also used for learning color models
% 2 -> pixels contrained to be background, these are also used for learning color models

img=imread('data/cows.jpg');
labels=imread('data/savedLabels/cows-labels.png');

% Make segmentation options
opts=bj.segOpts(); % You can use other methods such as gsc.opts() or gscSeq.opts();

% Customise the options if you need
opts.postProcess=1;
opts.gcGamma=150;

% Intialize the segmentation object 
segH=bj.segEngine(0,opts); % Again, you can use other methods such as gsc.segEngine() or
                           % gscSeq.segEngine(), but make sure you call the same
                           % package you used to build the options for

% preProcess image
segH.preProcess(im2double(img)); % Only takes in double images

% Get the first segmentation
ok=segH.start(labels);

% You can access the segmentation obtained using
figure;imshow(segH.seg);

% Delete the segmentation object
delete(segH);

% Another function that is more useful in an interactive setting is
% updateSeg -- for making edits to the obtained segmentation 
% you should call segH.updateSeg(newLabels) where newLabels would be the updated
% brush strokes. In an offline setting, you probably dont need to call updateSeg
