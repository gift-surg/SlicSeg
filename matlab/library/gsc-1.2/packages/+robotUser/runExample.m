function runExample()
% Code to run example robot user

cwd=miscFns.extractDirPath(mfilename('fullpath'));
inImg=[cwd '../../data/bike.jpg'];
inGT=[cwd '../../data/gt/bike.png'];
inLabels=[cwd '../../data/savedLabels/bike-labels.png'];

opts.brushStyle='dotMiddle';
opts.numStrokes=4;
opts.brushRad=8;

segOpts=gscSeq.segOpts();
segH=gscSeq.segEngine(0,segOpts);

[annoSeq,segSeq]=robotUser.run(segH,opts,im2double(imread(inImg)),imread(inGT),imread(inLabels));
delete(segH);

% --- Show the segmentation and brush sequence ---
figure;
for i=1:opts.numStrokes
  subplot(1,opts.numStrokes,i);
  imshow(segSeq(:,:,i));
  title(sprintf('Segmentation after\nstroke %d',i-1));
end

figure;
[xx,labelCmap]=imread(inLabels);
for i=1:opts.numStrokes
  subplot(1,opts.numStrokes,i);
  imshow(label2rgb(annoSeq(:,:,i)+1,labelCmap));
  title(sprintf('Brushes after\nstroke %d',i-1));
end

