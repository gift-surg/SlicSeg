function SavePNGSegmentation(segImage, segSaveFolder, orientation)
    % SavePNGSegmentation Saves segmentation as a series of PNG files
    % indexed from 1 to the number of slices.
    %
    % Author: Guotai Wang
    % Copyright (c) 2014-2016 University College London, United Kingdom. All rights reserved.
    % http://cmictig.cs.ucl.ac.uk
    %
    % Distributed under the BSD-3 licence. Please see the file licence.txt 
    % This software is not certified for clinical use.
    % 

    if nargin < 3
        orientation = 3;
    end
    
    for index = 1 : segImage.getMaxSliceNumber(orientation)
        segFileName = [segSaveFolder, num2str(index), '.png'];
        imwrite(segImage.rawImage(:,:,index)*255, segFileName);
    end
end