function SaveArrayToFile(filename,A)
fid=fopen(filename,'w');
[m,n]=size(A);
 for i=1:1:m
    for j=1:1:n
       if j==n
         fprintf(fid,'%g\n',A(i,j));
      else
        fprintf(fid,'%g\t',A(i,j));
       end
    end
end
fclose(fid);