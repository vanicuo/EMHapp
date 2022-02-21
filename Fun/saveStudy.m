% Save Study
function appTemp = saveStudy(appTemp)
    if(~exist(fullfile(appTemp.ProjectDir,appTemp.ProtocolName,appTemp.SubjName,appTemp.StudyName), 'file'))
        mkdir(fullfile(appTemp.ProjectDir,appTemp.ProtocolName,appTemp.SubjName,appTemp.StudyName));
    end
    % Save Spike View resluts
    if(appTemp.SaveState(3)==1)
        SaveSpikeViewResults = appTemp.SpikeView;
        if(isfield(SaveSpikeViewResults,'Event'))
            save(fullfile(appTemp.ProjectDir,appTemp.ProtocolName,appTemp.SubjName,appTemp.StudyName,'SaveSpikeViewResults.mat'),'-struct','SaveSpikeViewResults','-v7.3');
        end
    end          
    % Save HFO View resluts
    if(appTemp.SaveState(6)==1)
        SaveHFOViewResults = appTemp.HFOView;
        if(isfield(SaveHFOViewResults,'Event'))
            save(fullfile(appTemp.ProjectDir,appTemp.ProtocolName,appTemp.SubjName,appTemp.StudyName,'SaveHFOViewResults.mat'),'-struct','SaveHFOViewResults','-v7.3');
        end
    end  
    appTemp.SaveState = zeros(6,1);
end