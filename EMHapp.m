emhapp_path = strrep(which('EMHapp.m'), [filesep, 'EMHapp.m'], '');
addpath(genpath(fullfile(emhapp_path, 'Fun')));
addpath(genpath(fullfile(emhapp_path, 'GUI')));
run('Main')