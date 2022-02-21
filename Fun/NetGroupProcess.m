function appTemp = NetGroupProcess(temp)
    % Load Data
    Temp = load(temp);appTemp = Temp.appTemp;Group = Temp.Group;
    % Start
    appTemp = StartFcn(appTemp);
    % Group Process Data
    for subj=1:length(Group.GroupProcessStudy)
        try
            % Add Subject Information to app
            GroupProcessStudyTemp = Group.GroupProcessStudy{subj};
            appTemp = PrepareParamApp(GroupProcessStudyTemp, appTemp);

            % Load/Make Leadfield
            MRI = appTemp.Mri;VoxelSize = appTemp.VirtualSensorOpt.Volume_Resolutione / 1000;
            ChannelMats = appTemp.SutdiesInSubject.ChannelMat;
            FifFiles = cellfun(@(x)x.filename, appTemp.SutdiesInSubject.sFile, 'UniformOutput', false);
            OutDirs = cellfun(@(x)fullfile(appTemp.ProjectDir,appTemp.ProtocolName,appTemp.SubjName,x,'Ft_Process'), appTemp.SutdiesInSubject.StudyName, 'UniformOutput', false);
            [leadfields, bstChannel] = VolumeModel(MRI, VoxelSize, ChannelMats, FifFiles, OutDirs);

            % Prepare parameters to pipeline
            ParamPath = fullfile(appTemp.ProjectDir,appTemp.ProtocolName, 'Parameters.mat');
            Files.RawFile = FifFiles;Files.FileIED = strrep(FifFiles, 'MEG_ICA', 'MEG_IED');Files.FileHFO = strrep(FifFiles, 'MEG_ICA', 'MEG_HFO');
            Files.HFODetectionDirs = cellfun(@(x)fullfile(appTemp.ProjectDir,appTemp.ProtocolName,appTemp.SubjName,x,'SaveHFODetectionResults.mat'), appTemp.SutdiesInSubject.StudyName, 'UniformOutput', false);
            Files.VirtualSensorDirs = cellfun(@(x)fullfile(appTemp.ProjectDir,appTemp.ProtocolName,appTemp.SubjName,x,'SaveVirtualSensorResults.mat'), appTemp.SutdiesInSubject.StudyName, 'UniformOutput', false);
            PipelineParam.Files = Files;
            PipelineParam.PreprocessOpt = appTemp.PreprocessOpt;
            PipelineParam.IEDdetectionOpt = appTemp.IEDdetectionOpt;
            PipelineParam.VirtualSensorOpt = appTemp.VirtualSensorOpt;
            PipelineParam.HFOdetectionOpt = appTemp.HFOdetectionOpt;
            PipelineParam.bstChannel = bstChannel;
            PipelineParam.leadfields=leadfields;
            PipelineParam.cuda_device=appTemp.Device;
            save(ParamPath,'-struct','PipelineParam','-v6');

            % Call Python and Run
            tempDir = fullfile(appTemp.FunPath,'EMHapp');
            cmd = ['source ~/.bashrc;source activate;conda activate HFO;python ', fullfile(tempDir, 'emhapp_run.py'), ' --mat ', ParamPath];
            system(cmd);

            % Write HFOView mat Files
            for i=1:size(GroupProcessStudyTemp,1)
                appTemp.StudyName = appTemp.SutdiesInSubject.StudyName{i};
                appTemp.iStudy = appTemp.SutdiesInSubject.iStudy(i);
                appTemp.sFile = appTemp.SutdiesInSubject.sFile{i};
                appTemp.ChannelMat = appTemp.SutdiesInSubject.ChannelMat{i};
                appTemp.SaveState = appTemp.SutdiesInSubject.SaveState{i};
                if(exist(fullfile(appTemp.ProjectDir,appTemp.ProtocolName,appTemp.SubjName,appTemp.SutdiesInSubject.StudyName{i},'SaveHFODetectionResults.mat'), 'file'))
                    appTemp.HFOdetectionResults = load(fullfile(appTemp.ProjectDir,appTemp.ProtocolName,appTemp.SubjName,appTemp.SutdiesInSubject.StudyName{i},'SaveHFODetectionResults.mat'));
                    appTemp.VirtualSensorResults = load(fullfile(appTemp.ProjectDir,appTemp.ProtocolName,appTemp.SubjName,appTemp.SutdiesInSubject.StudyName{i},'SaveVirtualSensorResults.mat'));
                    appTemp = LoadHFOForView(appTemp);
                    appTemp.SaveState(6)=1;appTemp = saveStudy(appTemp);
                end
            end
        catch exception
            message = ['[ERROR][',exception.identifier,'][', datestr(clock), '][', Group.GroupProcessStudy{subj}{1, 2},  ']: ', exception.message];
            log_files = fullfile(appTemp.ProjectDir, 'NetRun', 'Logs', [Group.GroupProcessStudy{subj}{1, 2}, '.txt']);
            fid = fopen(log_files, 'w');
            fprintf(fid, '%s\n', message);
            fclose(fid);
        end
    end
end