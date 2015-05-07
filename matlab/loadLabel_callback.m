function loadLabel_callback(handles)
global ILabel;
global segSaveFolder;
global Strokefolder;
[labelFileName,labelFolderName,FilterIndex] = uigetfile('*.png');
if(labelFileName==0)
    return;
end

rgbLabel=imread(fullfile(labelFolderName,labelFileName));
ISize=size(rgbLabel);
ILabel=uint8(zeros(ISize(1),ISize(2)));
for i=1:ISize(1)
    for j=1:ISize(2)
        if(rgbLabel(i,j,1)==255 && rgbLabel(i,j,2)==0 && rgbLabel(i,j,3)==0)
            hold on;
            plot(j,i,'.r','MarkerSize',2);
            ILabel(i,j)=127;
        elseif(rgbLabel(i,j,1)==0 && rgbLabel(i,j,2)==0 && rgbLabel(i,j,3)==255)
            hold on;
            plot(j,i,'.b','MarkerSize',2);
            ILabel(i,j)=255;
        end
    end
end
findstroke=strfind(labelFolderName,'/stroke');
if(~isempty(findstroke))
    segSaveFolder=labelFolderName(1:findstroke-1);
    Strokefolder=labelFolderName;
else
    segSaveFolder=labelFolderName;
end