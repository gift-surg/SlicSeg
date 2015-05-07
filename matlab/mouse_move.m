function mouse_move(imagefig,varargins)
global mouse_state;
global for_back_ground;
global ILabel;
if(mouse_state==0 || for_back_ground==0)
    return;
end
radius=2;
temp = get(gca,'currentpoint');
x=floor(temp(1,2));
y=floor(temp(1,1));
if(for_back_ground==1)
    hold on;
    plot(temp(1,1),temp(1,2),'.r','MarkerSize',10);

    for i=-radius:radius
        for j=-radius:radius
            %if(i*i+j*j<=radius*radius)
            ILabel(x+i,y+j)=127;
            %end
        end
    end
else
    temp = get(gca,'currentpoint');
    hold on;
    plot(temp(1,1),temp(1,2),'.b','MarkerSize',10);
    for i=-radius:radius
        for j=-radius:radius
            %if(i*i+j*j<=radius*radius)
            ILabel(x+i,y+j)=255;
            %end
        end
    end
end
