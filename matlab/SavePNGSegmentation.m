function SavePNGSegmentation(obj, segImage, segSaveFolder)
    for index = 1 : obj.segImage.ImageSize(obj.orientation)
        segFileName=fullfile(segSaveFolder,[num2str(index) '_seg.png']);
        imwrite(segImage.RawImage(:,:,index)*255,segFileName);
    end
end