function ShowLocation(app)
    clear global hdl
    global hdl
    hdl.app = app;
    EventIdx = cellfun(@(x)strcmp(x,app.AllHFOView.Event.EventValue),app.AllHFOView.Event.EventName);
    hdl.Loc = [];hdl.ChanNum = [];hdl.SeegChannelMat = [];
    hdl.bstChannel=app.AllHFOView.toShow.ChannelMat_VS(EventIdx);
    % Show Figure
    % Add private figure_3d.m Path
    PrivateFigure3dPath = which('Main.mlapp');
    addpath([fileparts(PrivateFigure3dPath),'/ExternalFun']);
    hdl.SurfPos = [0.00 0.000 0.80 1.00];
    [hdl.Fig, ~, ~] = view_surface(hdl.app.ProtocolFile.ProtocolSubjects(hdl.app.iSubject).Subject.Surface(6).FileName,0.5);
    rmpath([fileparts(PrivateFigure3dPath),'/ExternalFun']);
    panel_scout('SetCurrentAtlas',1);
    set(hdl.Fig,'Name','Location','NumberTitle', 'off');
    hdl.axes = findobj(hdl.Fig, '-depth', 2, 'tag', 'Axes3D');
    % add pannel
    hdl.panel3 = uipanel('Parent',hdl.Fig,'Position',[0.80 0 0.20 1],'Units','normalized');
    % Pannel3
    Item = cat(2,app.ProtocolFile.ProtocolStudies(app.iSubject).Study.Condition);
    Item = cat(2,'None',Item(contains(Item,'Implantation')),Item(~contains(Item,'Implantation')));
    hdl.pop = uicontrol('Style', 'popup','String', Item,'Parent',hdl.panel3,'Units','normalized','Position', [0 0.85 1 0.1],'Min',1,'Max',length(Item));
    hdl.list = uicontrol('Style','listbox','Parent',hdl.panel3,'Units','normalized','String',{},'position',[0.05 0.05 0.90 0.78]);   
    hdl.pop.Callback = @CallPop;
    hdl.list.Callback = @CallList;
    % Show VS
    AxesPos = hdl.axes.Position(3)*hdl.Fig.Position(3);hdl.LabelSize = round(AxesPos/20);
    HFOIdx = hdl.app.AllHFOView.HFOIdx{EventIdx};
    ChanLoc = cat(2,hdl.bstChannel.Channel.Loc);
    Loc = ChanLoc(:,HFOIdx(hdl.app.AllHFOView.HFOIdxToShow{EventIdx},2));
    hold(hdl.axes,'on');plot3(hdl.axes,Loc(1,:),Loc(2,:),Loc(3,:),'.r','MarkerSize',hdl.LabelSize);hold(hdl.axes,'off');
    hdl.Fig.SizeChangedFcn = @CallFunSizeChange;
end

function CallFunSizeChange(hObject,eventdata,handles)
    global hdl
    AxesPos = hdl.axes.Position(3)*hdl.Fig.Position(3);hdl.LabelSize = round(AxesPos/20);
    set(hdl.axes.Children(ismember(hdl.axes.Children.get('Tag'),'VS')),'SizeData',hdl.LabelSize*50);
    set(hdl.axes.Children(ismember(hdl.axes.Children.get('Tag'),'SEEG')),'MarkerSize',hdl.LabelSize);
    set(hdl.axes.Children(ismember(hdl.axes.Children.get('Tag'),'SEEGi')),'LineWidth',hdl.LabelSize/8);
    temp = hdl.axes.Children(contains(hdl.axes.Children.get('Type'),'text'));
    set(temp(ismember(temp.get('Tag'),'SEEGiTag')),'FontSize',hdl.LabelSize/1.5);
    set(temp(ismember(temp.get('Tag'),'SEEGTag')),'FontSize',hdl.LabelSize/4);
end

function CallPop(hObject,eventdata,handles)
    global hdl
    Value = hdl.pop.Value;
    try
        % Show event
        iStudy = cellfun(@(x)strcmp(hdl.pop.String{Value},x),{hdl.app.ProtocolFile.ProtocolStudies.Study.Condition});
        Protocol = bst_get('ProtocolInfo');hdl.SeegChannelMat = load(fullfile(Protocol.STUDIES,hdl.app.ProtocolFile.ProtocolStudies.Study(iStudy).Channel.FileName));
        channel = cellfun(@(x)[x,' '],{hdl.SeegChannelMat.Channel.Name},'UniformOutput',0);
        set(hdl.list,'String',sort(channel),'Max',length(channel),'Value',1);
    catch
        delete(hdl.axes.Children(contains(hdl.axes.Children.get('Tag'),'SEEG')));
        delete(hdl.axes.Children(contains(hdl.axes.Children.get('Type'),'text')));
        set(hdl.list,'String',{});
    end
end

function CallList(hObject,eventdata,handles)
    global hdl
    Value = hdl.pop.Value;
    try
        % Show SEEG postionx(:,2)
        iStudy = cellfun(@(x)strcmp(hdl.pop.String{Value},x),{hdl.app.ProtocolFile.ProtocolStudies(hdl.app.iSubject).Study.Condition});
        Protocol = bst_get('ProtocolInfo');hdl.SeegChannelMat = load(fullfile(Protocol.STUDIES,hdl.app.ProtocolFile.ProtocolStudies(hdl.app.iSubject).Study(iStudy).Channel.FileName));  
        delete(hdl.axes.Children(contains(hdl.axes.Children.get('Tag'),'SEEG')));
        delete(hdl.axes.Children(contains(hdl.axes.Children.get('Type'),'text')));
        Value = hdl.list.Value;channel = sort(cellfun(@(x)[x,' '],{hdl.SeegChannelMat.Channel.Name},'UniformOutput',0));Value = cellfun(@(x)find(ismember({hdl.SeegChannelMat.Channel.Name},x(1:end-1))),channel(Value));
        Pos = cat(2,hdl.SeegChannelMat.Channel(Value).Loc);Name = {hdl.SeegChannelMat.Channel(Value).Name};
        hold(hdl.axes,'on');plot3(hdl.axes,Pos(1,:),Pos(2,:),Pos(3,:),'.g','MarkerSize',hdl.LabelSize,'Tag','SEEG');
        if(~isempty(hdl.SeegChannelMat.IntraElectrodes))
            cellfun(@(x)plot3(hdl.axes,x(1,:),x(2,:),x(3,:),'b','LineWidth',hdl.LabelSize/8,'Tag','SEEGi'),{hdl.SeegChannelMat.IntraElectrodes(cellfun(@(x)any(ismember(unique({hdl.SeegChannelMat.Channel(Value).Group}),x)),{hdl.SeegChannelMat.IntraElectrodes.Name})).Loc});hold(hdl.axes,'off');
            Group = unique({hdl.SeegChannelMat.Channel(Value).Group},'stable');
            temp = {hdl.SeegChannelMat.IntraElectrodes(cellfun(@(x)any(ismember(unique({hdl.SeegChannelMat.Channel(Value).Group}),x)),{hdl.SeegChannelMat.IntraElectrodes.Name})).Loc};
            temp = cell2mat(cellfun(@(x)x(:,2),temp,'UniformOutput',false));
            text(hdl.axes,temp(1,:),temp(2,:),temp(3,:)+0.002,Group,'color','w','FontSize',hdl.LabelSize/1.5,'FontWeight','bold','Tag','SEEGiTag');
        end
        text(hdl.axes,Pos(1,:),Pos(2,:),Pos(3,:)+0.001,Name,'color','w','FontSize',hdl.LabelSize/4,'FontWeight','bold','Tag','SEEGTag');
    catch 
        delete(hdl.axes.Children(contains(hdl.axes.Children.get('Tag'),'SEEG')));
        delete(hdl.axes.Children(contains(hdl.axes.Children.get('Type'),'text')));
    end
end