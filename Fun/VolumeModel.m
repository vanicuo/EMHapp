function [leadfields, bstChannel] = VolumeModel(MRI, VoxelSize, ChannelMats, FifFiles, OutDirs)
    %% ========Make Source Model=======
    if(exist(fullfile(OutDirs{1},'LeadField_Volume.mat'), 'file') && ~isempty(whos('seg','-file', fullfile(OutDirs{1},'LeadField_Volume.mat'))))
        temp = load(fullfile(OutDirs{1},'LeadField_Volume.mat'));
        seg = temp.seg;brain_mesh = temp.brain_mesh;ftSourcemodel = temp.ftSourcemodel;Channel = temp.Channel;bstChannel = temp.bstChannel;
    else
        [seg, brain_mesh, ftSourcemodel, Channel, bstChannel] = MakeSourceModel(MRI, VoxelSize);
    end
    
    %% =========Make Leadfield=========
    leadfields = cell(length(ChannelMats), 1);
    for i=1:length(ChannelMats)
        OutDir = OutDirs{i};
        if(exist(fullfile(OutDir,'LeadField_Volume.mat'), 'file') && ~isempty(whos('seg','-file', fullfile(OutDir,'LeadField_Volume.mat'))))
            temp = load(fullfile(OutDir,'LeadField_Volume.mat'), 'ftLeadfield');ftLeadfield = temp.ftLeadfield;
            leadfields{i} = permute(cat(3,ftLeadfield.leadfield{find(ftLeadfield.inside)}),[3,2,1]);
        else
            ChannelMat = ChannelMats{i};
            FifFile = FifFiles{i};
            % Covert Channel
            [~, ftGrad] = out_fieldtrip_channel(ChannelMat);
            ftGrad = ft_convert_units(ftGrad,'m');
            sens = ft_read_sens(FifFile, 'senstype', 'meg');
            sens = ft_convert_units(sens,'m');
            [~, M] = ft_warp_optim(sens.chanpos, ftGrad.chanpos,'rigidbody');
            M=rigidbody(M);
            ftGrad=ft_transform_sens(M,sens);
            % Build Headmodel
            cfg = [];
            cfg.feedback=0;
            cfg.grad = ftGrad;
            cfg.method = 'localspheres';
            cfg.unit = 'm';
            ftHeadmodel = ft_prepare_headmodel(cfg, brain_mesh);
            % Build Leadfield 
            cfg = [];
            cfg.channel = 'MEG';
            cfg.headmodel = ftHeadmodel;
            cfg.grad = ftGrad;
            cfg.grid = ftSourcemodel;
            cfg.normalize = 'yes';
            cfg.reducerank = 2;
            ftLeadfield = ft_prepare_leadfield(cfg);
            % Modify ftLeadfield 
            ftLeadfield.inside=zeros(length(ftLeadfield.inside),1);
            ftLeadfield.inside([Channel{:,1}])=1;  
            % Export leadfield
            leadfields{i} = permute(cat(3,ftLeadfield.leadfield{find(ftLeadfield.inside)}),[3,2,1]);
            % Save
            if(~exist(OutDir, 'dir'))
               mkdir(OutDir) 
            end
            save(fullfile(OutDir,'LeadField_Volume.mat'), 'seg', 'brain_mesh', 'ftSourcemodel', 'ftHeadmodel', 'ftLeadfield', 'Channel', 'bstChannel');
        end
    end
    leadfields = permute(cat(4, leadfields{:}), [4, 1, 2, 3]);
end