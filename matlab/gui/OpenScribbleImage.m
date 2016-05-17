 function seedImage = OpenScribbleImage(labelFileName)
    % SavePNGSegmentation Eead scribbles in the start slice (*.png rgb file)
    % indexed from 1 to the number of slices.
    %
    % Author: Guotai Wang
    % Copyright (c) 2014-2016 University College London, United Kingdom. All rights reserved.
    % http://cmictig.cs.ucl.ac.uk
    %
    % Distributed under the BSD-3 licence. Please see the file licence.txt 
    % This software is not certified for clinical use.
    %  

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
