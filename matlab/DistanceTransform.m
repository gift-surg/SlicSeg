function DF=DistanceTransform(I)
%input: segmentation result
[M,N]=size(I);
veryLarge=1e5;
DF=veryLarge*ones([M,N]);
for i=1:M
    for j=1:N
        if(I(i,j)==1)
            if(~(I(i-1,j)==0 || I(i+1,j)==0 || I(i,j-1)==0 || I(i,j+1)==0))
                DF(i,j)=0;
            end
        end
    end
end

% P=double(P)/255;
for i=2:M
    for j=2:N
        temp_up=DF(i-1,j)+1;
        temp_left=DF(i,j-1)+1;
        DF(i,j)=min([DF(i,j) temp_up temp_left]);
    end
end

for i=M:-1:1
    for j=N:-1:1
        if(i==M && j<N)
            DF(i,j)=min(DF(i,j),DF(i,j+1)+1);
        elseif(i<M && j==N)
            DF(i,j)=min(DF(i,j),DF(i+1,j)+1);
        elseif(i<M && j<N)
            temp_down=DF(i+1,j)+1;
            temp_right=DF(i,j+1)+1;
            DF(i,j)=min([DF(i,j) temp_down temp_right]);
        end
    end
end

for i=2:M
    for j=2:N
        temp_up=DF(i-1,j)+1;
        temp_left=DF(i,j-1)+1;
        DF(i,j)=min([DF(i,j) temp_up temp_left]);
    end
end

for i=M:-1:1
    for j=N:-1:1
        if(i==M && j<N)
            DF(i,j)=min(DF(i,j),DF(i,j+1)+1);
        elseif(i<M && j==N)
            DF(i,j)=min(DF(i,j),DF(i+1,j)+1);
        elseif(i<M && j<N)
            temp_down=DF(i+1,j)+1;
            temp_right=DF(i,j+1)+1;
            DF(i,j)=min([DF(i,j) temp_down temp_right]);
        end
    end
end