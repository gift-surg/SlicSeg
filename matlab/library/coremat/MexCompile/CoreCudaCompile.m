classdef CoreCudaCompile < handle
    % CoreCudaCompile. Class for compiling mex files
    %
    %
    %
    %     Licence
    %     -------
    %     Part of CoreMat. https://github.com/tomdoel/coremat
    %     Author: Tom Doel, 2013.  www.tomdoel.com
    %     Distributed under the MIT licence. Please see website for details.
    %    
    
    methods (Static)
        function mex_result = Compile(compiler, mex_file, src_fullfile, output_directory, host_compiler)
            compile_arguments = ['"' compiler '" -ptx --output-directory ' output_directory ' ' src_fullfile, ' ' mex_file.OtherCompilerFiles];
            if ~isempty(host_compiler)
                compile_arguments = [compile_arguments ' --compiler-bindir ' host_compiler];
            end
            mex_result = system(compile_arguments);
        end
    end
end

