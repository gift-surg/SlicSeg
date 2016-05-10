function SavePNGSegmentation(segImage, segSaveFolder, orientation)
    if nargin < 3
        orientation = 3;
    end
    for index = 1 : segImage.getMaxSliceNumber(orientation)
        segFileName=fullfile(segSaveFolder,[num2str(index) '_seg.png']);
        imwrite(segImage.RawImage(:,:,index)*255,segFileName);
    end
end