classdef SlicSegGuiController < handle
    
    properties (SetAccess = private)
        ILabel
    end
    
    properties (Access = private)
        mouseDown = false
        foreground = true
    end
    
    methods
        function obj = SlicSegGuiController()
        end
        
        function SelectForeground(obj)
            obj.foreground = true;
        end
        
        function selectForeground(obj)
            obj.foreground = false;
        end
        
        function ResetLabelImage(obj, imgSize)
            obj.ILabel = uint8(zeros([imgSize(1), imgSize(2)]));
        end
        
        function mouse_down(obj, imagefig, varargins)
            obj.mouseDown = true;
        end
        
        function mouse_up(obj, imagefig, varargins)
            obj.mouseDown = false;
        end
        
        function mouse_move(obj, imagefig, varargins)
            if(~obj.mouseDown)
                return;
            end
            radius=2;
            temp = get(gca, 'currentpoint');
            x=floor(temp(1,2));
            y=floor(temp(1,1));
            if(obj.foreground)
                hold on;
                plot(temp(1,1),temp(1,2),'.r','MarkerSize',10);
                
                for i=-radius:radius
                    for j=-radius:radius
                        obj.ILabel(x+i,y+j)=127;
                    end
                end
            else
                temp = get(gca,'currentpoint');
                hold on;
                plot(temp(1,1),temp(1,2),'.b','MarkerSize',10);
                for i=-radius:radius
                    for j=-radius:radius
                        obj.ILabel(x+i,y+j)=255;
                    end
                end
            end
        end
    end
end

