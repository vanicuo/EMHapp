function appTemp = LoadHFOForView(appTemp)
    if(isfield(appTemp.HFOdetectionResults, 'EntHFO')&&~isempty(appTemp.HFOdetectionResults.EntHFO))
        % Load data
        VS_ChannelMat = appTemp.VirtualSensorResults.bstChannel;
        % Export to global value
        % For Show (BaseLine MEG sample, VS Weight, Channel Name and HFO time in BaseLine)
        appTemp.HFOView.toShow.ReSample = 1000;
        appTemp.HFOView.toShow.BsTime = appTemp.HFOdetectionResults.EntHFO_BsTime;
        appTemp.HFOView.toShow.ArdBsTime = appTemp.HFOdetectionResults.ArdEntHFO_BsTime;
        appTemp.HFOView.toShow.Weight = appTemp.HFOdetectionResults.EntHFO_Weight;
        appTemp.HFOView.toShow.ArdWeight = appTemp.HFOdetectionResults.ArdEntHFO_Weight;
        appTemp.HFOView.toShow.Weight_S = appTemp.HFOdetectionResults.EntHFO_Weight_S;
        appTemp.HFOView.toShow.ArdWeight_S = appTemp.HFOdetectionResults.ArdEntHFO_Weight_S;
        appTemp.HFOView.toShow.Channel = cellfun(@(x)reshape(cat(1,{VS_ChannelMat.Channel(x(:,2)).Name},{VS_ChannelMat.Channel(x(:,2)).Group}), [], 1)',appTemp.HFOdetectionResults.EntHFO,'UniformOutput',false);
        appTemp.HFOView.toShow.ArdChannel = cellfun(@(x)reshape(cat(1,{VS_ChannelMat.Channel(x(:,2)).Name},{VS_ChannelMat.Channel(x(:,2)).Group}), [], 1)',appTemp.HFOdetectionResults.ArdEntHFO,'UniformOutput',false);
        appTemp.HFOView.toShow.EventTime = cellfun(@(x)round([min(x(:,3)),max(x(:,4))]),appTemp.HFOdetectionResults.EntHFO,'UniformOutput',false);
        % For Feature
        appTemp.HFOView.toShow.LL_HFO = appTemp.HFOdetectionResults.Features.LL_HFO;
        appTemp.HFOView.toShow.HilAmp_HFO = appTemp.HFOdetectionResults.Features.HilAmp_HFO;
        appTemp.HFOView.toShow.TFEntropy_HFO = appTemp.HFOdetectionResults.Features.TFEntropy_HFO;
        appTemp.HFOView.toShow.LL_Spike = appTemp.HFOdetectionResults.Features.LL_Spike;
        appTemp.HFOView.toShow.PeakAmp_Spike = appTemp.HFOdetectionResults.Features.PeakAmp_Spike;
        appTemp.HFOView.toShow.TFEntropy_Spike = appTemp.HFOdetectionResults.Features.TFEntropy_Spike;
        % For Cluster
        appTemp.HFOView.toShow.Bic = appTemp.HFOdetectionResults.Bic;
        appTemp.HFOView.toShow.ClsIdx = appTemp.HFOdetectionResults.ClsIdx;
        appTemp.HFOView.toShow.Cls = appTemp.HFOdetectionResults.Cls;
        appTemp.HFOView.toShow.Best_Cls = appTemp.HFOdetectionResults.Best_Cls;
        % For Event
        SampleEventLabel = cellfun(@(x)cat(2,'Event_',num2str(x)),num2cell(1:size(appTemp.HFOdetectionResults.EntHFO,2)),'UniformOutput',false);
        appTemp.HFOView.Event.EventName = SampleEventLabel;
        appTemp.HFOView.Event.EventValue = SampleEventLabel{1};
        appTemp.HFOView.Event.EventToDisp = true(length(SampleEventLabel),1);
        % For HFO Index
        appTemp.HFOView.HFOIdx = appTemp.HFOdetectionResults.EntHFO;
        appTemp.HFOView.ArdHFOIdx = appTemp.HFOdetectionResults.ArdEntHFO;
        appTemp.HFOView.HFOIdxToShow = 1 : size(appTemp.HFOdetectionResults.EntHFO, 2);
        appTemp.HFOView.SourceMaps = appTemp.HFOdetectionResults.SourceMapsHFO;
    else
        appTemp.HFOView.toShow = [];
        appTemp.HFOView.Event = [];
        appTemp.HFOView.Event.EventName={'none'};
        appTemp.HFOView.HFOIdx = [];
        appTemp.HFOView.Signal = [];
    end
end
