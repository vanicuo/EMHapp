function app = PrepareParamApp(GroupProcessStudy, app)
    Protocols = load(bst_get('BrainstormDbFile'));
    app.iProtocol = find(cellfun(@(x)strcmp(x,GroupProcessStudy{1,3}),{Protocols.ProtocolsListInfo.Comment}));
    gui_brainstorm('SetCurrentProtocol',app.iProtocol);
    app.ProtocolFile = load(fullfile(Protocols.ProtocolsListInfo(cellfun(@(x)strcmp(x,GroupProcessStudy{1,3}),{Protocols.ProtocolsListInfo.Comment})).STUDIES,'protocol.mat'));
    app.ProtocolName = Protocols.ProtocolsListInfo(app.iProtocol).Comment;
    % Read Subject Name
    app.SubjName = GroupProcessStudy{1,2};
    app.iSubject = find(cellfun(@(x)strcmp(x,app.SubjName),{app.ProtocolFile.ProtocolSubjects.Subject.Name}));
    Subject = app.ProtocolFile.ProtocolSubjects.Subject(app.iSubject);
    app.Mri = load(fullfile(Protocols.ProtocolsListInfo(app.iProtocol).SUBJECTS,Subject.Anatomy(Subject.iAnatomy).FileName));app.Mri.InitTransf = [];
    % Read Study Name
    AllStudies = app.ProtocolFile.ProtocolStudies.Study;
    app.SutdiesInSubject.StudyName = cell(size(GroupProcessStudy,1), 1);
    app.SutdiesInSubject.iStudy = zeros(size(GroupProcessStudy,1), 1);
    app.SutdiesInSubject.sFile = cell(size(GroupProcessStudy,1), 1);
    app.SutdiesInSubject.ChannelMat = cell(size(GroupProcessStudy,1), 1);
    app.SutdiesInSubject.SaveState = cell(size(GroupProcessStudy,1), 1);
    for i=1:size(GroupProcessStudy,1)
        % init value, using the iProtocol/iSubject/iStudy for GroupDialog
        app.SutdiesInSubject.StudyName{i} = GroupProcessStudy{i,1};
        app.SutdiesInSubject.iStudy(i) = find(cellfun(@(x)strcmp(x,app.SutdiesInSubject.StudyName{i}),cat(1,app.ProtocolFile.ProtocolStudies.Study.Condition)));
        app.SutdiesInSubject.sFile{i} = load(fullfile(Protocols.ProtocolsListInfo(app.iProtocol).STUDIES,AllStudies(app.SutdiesInSubject.iStudy(i)).Data.FileName));
        app.SutdiesInSubject.sFile{i} = app.SutdiesInSubject.sFile{i}.F;
        app.SutdiesInSubject.ChannelMat{i} = load(fullfile(Protocols.ProtocolsListInfo(app.iProtocol).STUDIES,AllStudies(app.SutdiesInSubject.iStudy(i)).Channel.FileName));
        app.SutdiesInSubject.SaveState{i} = zeros(6,1);
        if(~exist(fullfile(app.ProjectDir,app.ProtocolName,app.SubjName,app.SutdiesInSubject.StudyName{i}), 'file'))
            mkdir(fullfile(app.ProjectDir,app.ProtocolName,app.SubjName,app.SutdiesInSubject.StudyName{i}));
        end
    end
end