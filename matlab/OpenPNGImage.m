function loadedImage = OpenImage(imgFolderName)
    % read volume image from a folder, which contains a chain of
    % *.png images indexed from 1 to the number of slices.

    dirinfo=dir(fullfile(imgFolderName,'*.png'));
    sliceNumber=length(dirinfo);
    
    longfilename=fullfile(imgFolderName,'1.png');
    I=imread(longfilename);
    size2d=size(I);
    size3d=[size2d, sliceNumber];
    volume=uint8(zeros(size3d));
    for i=1:sliceNumber
        tempfilename=fullfile(imgFolderName,[num2str(i) '.png']);
        tempI=imread(tempfilename);
        volume(:,:,i)=tempI(:,:);
    end
    loadedImage = volume;
end
