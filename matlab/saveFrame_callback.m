function saveFrame_callback(handles,startslice,newtrial)
global lambda;
global onlineP;
global offlineP;
global seedsRGB;
global currentSegLabel;
global segRGB;
global offSegLabel;
global offSegRGB;
global gcSegLabel;
global gcSegRGB;
global geosSegLabel;
global geosSegRGB;
global currentImageIndex;
global startImageIndex;
global segSaveFolder;
global SegEnable;
global OffSegEnable;
global GeoSEnable;
global GCEnable;
if((startslice && isempty(segSaveFolder)) || newtrial)
    segSaveFolder=getSaveFolderName(handles);
end

if(~isempty(segSaveFolder))
    if(currentImageIndex==startImageIndex)
        seedsrgbFileName=fullfile(segSaveFolder,[num2str(currentImageIndex) '_seedsrgb.png']);
        imwrite(seedsRGB,seedsrgbFileName);
    end
    username=get(handles.edit_username,'String');
    postfix='.png';
    if(username=='lambda')
        postfix=['lambda' num2str(lambda) '.png']
    end
    if(SegEnable)
        possibilityFileName=fullfile(segSaveFolder,[num2str(currentImageIndex) '_prob.png']);
        imwrite(onlineP,possibilityFileName);

        segFileName=fullfile(segSaveFolder,[num2str(currentImageIndex) '_seg' postfix]);
        imwrite(currentSegLabel*255,segFileName);
        segrgbFileName=fullfile(segSaveFolder,[num2str(currentImageIndex) '_segrgb' postfix]);
        imwrite(segRGB,segrgbFileName);
    end

    if(OffSegEnable)
        possibilityFileName=fullfile(segSaveFolder,[num2str(currentImageIndex) '_proboff.png']);
        imwrite(offlineP,possibilityFileName);
        segFileName=fullfile(segSaveFolder,[num2str(currentImageIndex) '_segoff.png']);
        imwrite(offSegLabel*255,segFileName);
        segrgbFileName=fullfile(segSaveFolder,[num2str(currentImageIndex) '_segoffrgb.png']);
        imwrite(offSegRGB,segrgbFileName);
    end
    if(GeoSEnable)
        tempsegFileName=fullfile(segSaveFolder,[num2str(currentImageIndex) '_seggeos.png']);
        imwrite(geosSegLabel*255,tempsegFileName);
        tempsegrgbFileName=fullfile(segSaveFolder,[num2str(currentImageIndex) '_seggeosrgb.png']);
        imwrite(geosSegRGB,tempsegrgbFileName);
    end
    if(GCEnable)
        tempsegFileName=fullfile(segSaveFolder,[num2str(currentImageIndex) '_seggc.png']);
        imwrite(gcSegLabel*255,tempsegFileName);
        tempsegrgbFileName=fullfile(segSaveFolder,[num2str(currentImageIndex) '_seggcrgb.png']);
        imwrite(gcSegRGB,tempsegrgbFileName);
    end
    disp(['save result in:' segSaveFolder]);
end