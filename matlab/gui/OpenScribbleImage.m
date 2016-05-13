 function seedImage = OpenScribbleImage(labelFileName)
    % read scribbles in the start slice (*.png rgb file)
    rgbLabel=imread(labelFileName);
    ISize=size(rgbLabel);
    ILabel=uint8(zeros(ISize(1),ISize(2)));
    for i=1:ISize(1)
        for j=1:ISize(2)
            if(rgbLabel(i,j,1)==255 && rgbLabel(i,j,2)==0 && rgbLabel(i,j,3)==0)
                ILabel(i,j)=127;
            elseif(rgbLabel(i,j,1)==0 && rgbLabel(i,j,2)==0 && rgbLabel(i,j,3)==255)
                ILabel(i,j)=255;
            end
        end
    end
    seedImage = ILabel;
 end
