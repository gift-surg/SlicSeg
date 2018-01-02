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
        function mex_result = Compile(compiler, cpp_compiler, mex_file, src_fullfile, output_directory)
            if isempty(cpp_compiler)
                compile_arguments = ['"' compiler '" -ptx --output-directory ' output_directory ' ' src_fullfile, ' ' mex_file.OtherCompilerFiles];
            else
                compile_arguments = ['"' compiler '" -ptx --compiler-bindir=' cpp_compiler ' --output-directory ' output_directory ' ' src_fullfile, ' ' mex_file.OtherCompilerFiles];
            end
            mex_result = system(compile_arguments);
        end
    end
end

