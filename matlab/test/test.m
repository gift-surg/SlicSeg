clear;
slicSeg=SlicSegAlgorithm;
slicSeg.lambda = 5.0;
slicSeg.sigma = 3.5;
slicSeg.innerDis = 6;
slicSeg.outerDis = 6;
slicSeg.startIndex = 22;
imageFilepath = mfilename('fullpath');
[imageFilepath, ~, ~] = fileparts(imageFilepath);
seedImage = OpenScribbleImage(fullfile(imageFilepath, 'a23_05', '22_seedsrgb.png'));
slicSeg.volumeImage = double(OpenPNGImage(fullfile(imageFilepath, 'a23_05', 'img')));
slicSeg.seedImage.replaceImageSlice(seedImage, slicSeg.startIndex, 3);
slicSeg.sliceRange = [5,38];
slicSeg.RunSegmention();
tempDir = fullfile(imageFilepath, 'a23_05', 'seg');
if(~isdir(tempDir))  
    mkdir(tempDir);
end
SavePNGSegmentation(slicSeg.segImage, tempDir, 3);