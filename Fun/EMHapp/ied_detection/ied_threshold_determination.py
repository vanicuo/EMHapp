# -*- coding:utf-8 -*-
# @Time    : 2022/2/4
# @Author  : cuiwei
# @File    : ied_threshold_determination.py
# @Software: PyCharm
# @Script to:
#   - 确定阈值法检测ied中，使用的阈值大小

import os
import pandas as pd
import mne
import numpy as np
import torch
from . import ied_peak_feature
from . import ied_temporal_clustering
from . import ied_detection_threshold
import matplotlib.pyplot as plt

gpu_device = 2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)


def hilbert(Sig, H):
    Sig = torch.fft.fft(Sig, axis=-1)
    return torch.abs(torch.fft.ifft(Sig * H, axis=-1))


def hilbert_H(N):
    h = torch.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    return h


def annotate_bad_segment(raw_data, raw_info, Thre=5, min_length_good=0.2, device=torch.device('cpu')):
    # jump artifact detection
    data = torch.tensor(raw_data.copy()).to(device)
    # seg data into 10ms and 5ms overlap segment
    seg_idx = (torch.arange(0, data.shape[1], int(raw_info['sfreq'] * 0.01 * 0.5)).unsqueeze(0) +
               torch.arange(0, int(raw_info['sfreq'] * 0.01)).unsqueeze(1))[:, :-1]
    seg_idx = seg_idx[:, torch.where(seg_idx[-1, :] < data.shape[-1])[0]]
    seg_data = data[:, seg_idx.T].permute(1, 0, 2)
    # get absolute peak to peak in each segment, and extract mean value of top 5 channel.
    seg_data_peak2peak = (seg_data.amax(dim=-1) - seg_data.amin(dim=-1)).abs()
    seg_data_peak2peak = (seg_data_peak2peak - seg_data_peak2peak.mean(dim=0, keepdim=True)) / \
                         seg_data_peak2peak.std(dim=0, keepdim=True)
    seg_data_peak2peak = seg_data_peak2peak.topk(5, dim=-1)[0].mean(dim=-1)
    # get jump segment with peak2peak over threshold
    jump_seg = seg_idx[:, torch.where(seg_data_peak2peak > 15)[0]].reshape(-1)
    if len(jump_seg) > 0:
        jump_seg = (jump_seg + torch.arange(-0.2 * raw_info['sfreq'],
                                            0.2 * raw_info['sfreq']).unsqueeze(-1)).unique().long()
        # merge jump segments with interleave samples less than min_length_good
        min_samps = min_length_good * raw_info['sfreq']
        jump_seg = torch.cat([torch.tensor(range(jump_seg[x], jump_seg[x + 1]))
                              for x in torch.where((jump_seg[1:] - jump_seg[:-1] <= min_samps) &
                                                   (jump_seg[1:] - jump_seg[:-1] > 1))[0]] + [jump_seg])
        jump_seg = jump_seg[(jump_seg >= 0) & (jump_seg < data.shape[-1])]

    # artifact detection
    data = torch.tensor(mne.filter.filter_data(raw_data, raw_info['sfreq'], 40, None,
                                               verbose=False, n_jobs='cuda')).to(device)
    hilbert_h = hilbert_H(data.shape[1]).to(device)
    data = torch.cat([hilbert(x.reshape(1, -1), hilbert_h) for x in data])
    dataMag = data[torch.tensor(mne.pick_types(raw_info, meg='mag')).to(device)]
    dataGrad = data[torch.tensor(mne.pick_types(raw_info, meg='grad')).to(device)]
    dataMag = (dataMag - dataMag.mean(dim=1, keepdim=True)) / dataMag.std(dim=1, keepdim=True)
    dataGrad = (dataGrad - dataGrad.mean(dim=1, keepdim=True)) / dataGrad.std(dim=1, keepdim=True)
    art_scores_mag = dataMag.sum(axis=0) / np.sqrt(dataMag.shape[0])
    art_scores_mag = torch.tensor(
        mne.filter.filter_data(art_scores_mag.cpu().double(), raw_info['sfreq'], None, 4, verbose=False,
                               n_jobs='cuda')).to(device)
    art_scores_grad = dataGrad.sum(axis=0) / np.sqrt(dataGrad.shape[0])
    art_scores_grad = torch.tensor(
        mne.filter.filter_data(art_scores_grad.cpu().double(), raw_info['sfreq'], None, 4, verbose=False,
                               n_jobs='cuda')).to(device)

    # get mask
    min_samps = min_length_good * raw_info['sfreq']
    art_mask = (art_scores_mag > Thre) | (art_scores_grad > Thre)
    Temp = torch.where(art_mask)[0]
    Temp = [torch.tensor(range(Temp[x], Temp[x + 1]))
            for x in torch.where((Temp[1:] - Temp[:-1] <= min_samps) & (Temp[1:] - Temp[:-1] > 1))[0]]
    if len(Temp) > 0:
        art_mask[torch.cat(Temp)] = True
    # Exclude the bad segment with sample less than 0.3s (Duration of IEDs less than 200ms)
    Temp = torch.tensor([0.] + art_mask.float().tolist() + [0.]).diff()
    Temp = torch.stack([torch.where(Temp == 1)[0], torch.where(Temp == -1)[0]])
    Temp = [torch.arange(Temp[0, x], Temp[1, x])
            for x in torch.where(Temp.diff(dim=0) <= 0.3 * raw_info['sfreq'])[1]]
    if len(Temp) > 0:
        art_mask[torch.cat(Temp)] = False
    if len(jump_seg) > 0:
        art_mask[jump_seg] = False

    return art_mask.cpu().numpy().reshape(-1)


def plot_meg_data_in_ied_event(meg_data=None, figsize=(10, 25.6), dpi=75):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    # plot raw
    for i in range(3):
        ax = plt.Axes(fig, [1 / 3 * i, 0.0, 0.95 / 3, 0.96])
        ax.set_axis_off()
        fig.add_axes(ax)
        X = meg_data[torch.arange(i, 306, 3)]
        X = (X - X.amin(dim=-1, keepdim=True)) / (X.amax(dim=-1, keepdim=True) - X.amin(dim=-1, keepdim=True)) \
            + torch.arange(X.shape[0]).reshape(-1, 1)
        ax.plot(X.t(), 'b')
    plt.show()


if __name__ == "__main__":
    sub_name = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub06', 'sub09', 'sub10', 'sub11', 'sub12',
                'sub13', 'sub14', 'sub15', 'sub16', 'sub17', 'sub18', 'sub19', 'sub20', 'sub21', 'sub22',
                'sub23', 'sub24', 'sub25', 'sub26', 'sub27', 'sub28', 'sub29', 'sub30', 'sub31', 'sub32',
                'sub33', 'sub34', 'sub35', 'sub36', 'sub37']
    raw_data_path = '/data2/humanbrain/junxian/MEG_EMS/'
    ied_event_file = 'spike_306_newest_20201009.xlsx'

    # 读取每个IED的peaks
    fifs_all, ied_peaks_all, fit_goodness_all = [], [], []
    for subj in sub_name:
        xls = pd.read_excel(ied_event_file, sheet_name=subj)
        fit_goodness = torch.tensor(xls.values[:, np.where(np.array(xls.columns) == 'goodness')[0]].astype('float'))
        fit_goodness = fit_goodness[~fit_goodness.isnan()]
        ied_peaks = torch.tensor(xls.values[:, np.where(np.array(xls.columns) == 'Peak Time')[0]].astype('float'))
        ied_peaks = ied_peaks[~ied_peaks.isnan()]
        fifs = torch.tensor(xls.values[:, np.where(np.array(xls.columns) == 'EP')[0]].astype('float'))
        fifs = fifs[~fifs.isnan()].long()
        ied_peaks = [ied_peaks[torch.where(fifs == x)] for x in fifs.unique()]
        fit_goodness = [fit_goodness[torch.where(fifs == x)] for x in fifs.unique()]
        fifs = [os.path.join(raw_data_path, subj, subj + '_EP_' + str(x.tolist()) + '_tsss.fif') for x in fifs.unique()]
        fifs_all.append(fifs)
        ied_peaks_all.append(ied_peaks)
        fit_goodness_all.append(fit_goodness)

    # 计算每个患者的IED features
    sim_all, amplitude_all, half_slope_all = [], [], []
    for fifs, ied_peaks, fits_goodness in zip(fifs_all, ied_peaks_all, fit_goodness_all):
        ied_amplitude_peaks_all, ied_half_slope_peaks_all, data_segment_all = [], [], []
        for fif, ied_peak, fit_goodness in zip(fifs, ied_peaks, fits_goodness):
            if not os.path.isfile(fif):
                continue
            print(fif.split('/')[-1])
            raw = mne.io.read_raw_fif(fif, verbose='error', preload=True)
            raw.pick(mne.pick_types(raw.info, meg=True, ref_meg=False))
            bad_segment = annotate_bad_segment(raw.get_data(), raw.info, Thre=6, min_length_good=0.2,
                                               device=torch.device("cuda"))
            # Step1: 获取MEG信号中的peak
            find_peaks = PeakFeatures.CalPeakFeatures(raw_data=raw.get_data(), data_info=raw.info,
                                                      bad_segment=bad_segment, device=gpu_device)
            signal_peaks = find_peaks.cal_signal_peaks(smooth_windows=0.02, smooth_iterations=2)
            # Step2: 计算每个peak的特征
            peak_index = find_peaks.cal_peak_index(pos_peak_idx=signal_peaks['pos_peak_idx'],
                                                   neg_peak_idx=signal_peaks['neg_peak_idx'])
            peak_duration = find_peaks.cal_peak_duration(pos_peak_idx=signal_peaks['pos_peak_idx'],
                                                         neg_peak_idx=signal_peaks['neg_peak_idx'])
            peak_half_duration = \
                find_peaks.cal_half_peak_duration(pos_half_peak_idx_num=signal_peaks['pos_half_peak_idx_num'],
                                                  neg_half_peak_idx_num=signal_peaks['neg_half_peak_idx_num'])
            peak_amplitude = find_peaks.cal_peak_amplitude(pos_peak_data=signal_peaks['pos_peak_data'],
                                                           neg_peak_data=signal_peaks['neg_peak_data'])
            half_peak_slope = \
                find_peaks.cal_half_peak_slope(pos_peak_data=signal_peaks['pos_peak_data'],
                                               neg_peak_data=signal_peaks['neg_peak_data'],
                                               pos_half_peak_idx_num=signal_peaks['pos_half_peak_idx_num'],
                                               neg_half_peak_idx_num=signal_peaks['neg_half_peak_idx_num'])
            # Step3: 归一化每个peak的特征值
            peak_amplitude_zscore, peak_slope_zscore, half_peak_slope_zscore, peak_sharpness_zscore = \
                find_peaks.cal_zscored_features(peak_amplitude=peak_amplitude, half_peak_slope=half_peak_slope,
                                                peak_duration=peak_duration, exclude_duration=0.02,
                                                z_win=100, z_region=False, z_mag_grad=False)

            # Step4: 计算IED peak周围，每个通道的peak点位置
            ieds = GetIED.GetIED(raw_data=raw.get_data(), data_info=raw.info, device=gpu_device)
            samples_of_picked_peak = np.array([[-1]]).repeat(
                ieds.get_raw_data().shape[0], axis=0).repeat(ieds.get_raw_data().shape[1], axis=1)
            for i in range(samples_of_picked_peak.shape[0]):
                samples_of_picked_peak[i][peak_index[i].cpu()] = np.arange(peak_index[i].shape[0]).astype('long')
            ieds_peak_closest_idx = ieds.cal_peaks_closest_to_ieds_peaks(
                ieds_peak=(ied_peak * ieds.data_info['sfreq']).long(),
                ieds_win=torch.arange(-150, 150).unsqueeze(0).repeat(ied_peak.shape[0], 1).long(),
                samples_of_picked_peak=samples_of_picked_peak)

            # Step5: 获取数据片段
            data_segment = ieds.cal_data_segments_with_shift(data_segment_window=0.1,
                                                             ieds_peak_closest_idx=ieds_peak_closest_idx.clone())

            # Step6: 计算IED peak周围，每个通道的peak点位置，对应的特征大小
            ied_amplitude_peaks, ied_half_slope_peaks, ied_sharpness_peaks, ied_duration_peaks = \
                ieds.cal_ieds_peaks_features(samples_of_picked_peak=samples_of_picked_peak,
                                             ieds_peak_closest_idx=ieds_peak_closest_idx.clone(),
                                             peak_amplitude=peak_amplitude_zscore,
                                             half_peak_slope=half_peak_slope_zscore,
                                             peak_duration=peak_duration)
            ied_amplitude_peaks_all.append(ied_amplitude_peaks)
            ied_half_slope_peaks_all.append(ied_half_slope_peaks)
            data_segment_all.append(data_segment)

        ied_amplitude_peaks_all = torch.cat(ied_amplitude_peaks_all, dim=0)
        ied_half_slope_peaks_all = torch.cat(ied_half_slope_peaks_all, dim=0)
        data_segment_all = torch.cat(data_segment_all, dim=0)
        peaks_max_amplitude = ied_amplitude_peaks_all.argmax(dim=-1)
        peaks_amplitude = ied_amplitude_peaks_all.take_along_dim(peaks_max_amplitude.unsqueeze(-1), dim=-1).squeeze()
        peaks_half_slop = ied_half_slope_peaks_all.take_along_dim(peaks_max_amplitude.unsqueeze(-1), dim=-1).squeeze()
        peaks_amplitude_half_slop = (peaks_amplitude - peaks_amplitude.mean(dim=-1, keepdim=True)) / \
                                    peaks_amplitude.std(dim=-1, keepdim=True) + \
                                    (peaks_half_slop - peaks_half_slop.mean(dim=-1, keepdim=True)) / \
                                    peaks_half_slop.std(dim=-1, keepdim=True)
        channel_topk = peaks_amplitude_half_slop.topk(5, dim=-1)[1]
        channel_topk_amplitude = peaks_amplitude.take_along_dim(channel_topk, dim=-1)
        channel_topk_half_slope = peaks_half_slop.take_along_dim(channel_topk, dim=-1)
        data_segment_large_temp = data_segment_all.take_along_dim(channel_topk.unsqueeze(-1).unsqueeze(-1), dim=1)
        data_segment_large = data_segment_large_temp.reshape(-1, data_segment_large_temp.shape[-2],
                                                             data_segment_large_temp.shape[-1])
        # 获取前500个segment，防止显存溢出
        temp_index = peaks_amplitude_half_slop.take_along_dim(channel_topk, dim=-1).reshape(-1).sort()[1]
        temp_index = temp_index[-min(temp_index.shape[0], 1000):]
        data_segment_large = data_segment_large[temp_index]
        # Step7: 计算IED template
        temporal_cluster = TemporalCulster.TemporalCluster(device=gpu_device)
        clusters, clusters_data, clusters_center_data = \
            temporal_cluster.sequential_cluster_with_pearson_euler(
                data=data_segment_large.clone(), sim_threshold=0.85, dist_threshold=0.85, corr_threshold=0.85,
                number_threshold=5)
        # Step8: 计算相似度
        sim, corr, dist = \
            temporal_cluster.similarity_between_segments_templates(data_segment_large_temp,
                                                                   clusters_center_data.clone())
        peaks_sim = sim.amax(dim=-1)

        amplitude_all.append(channel_topk_amplitude)
        half_slope_all.append(channel_topk_half_slope)
        sim_all.append(peaks_sim)

    # 使用GOF
    if is_fog:
        fog = []
        for fifs, fits_goodness in zip(fifs_all, fit_goodness_all):
            for fif, fit_goodness in zip(fifs, fits_goodness):
                if not os.path.isfile(fif):
                    continue
                fog.append(fit_goodness)
        features = torch.stack([torch.cat(amplitude_all), torch.cat(half_slope_all),
                                torch.cat(sim_all)]).cpu().numpy()[:, fog > 0.75]
    else:
        features = torch.stack([torch.cat(amplitude_all), torch.cat(half_slope_all), torch.cat(sim_all)]).cpu().numpy()

    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    bins = 20
    fig = plt.figure(figsize=(20 / 1.2, 6 / 1.2), dpi=150)
    axes = plt.subplot(131)
    temp = features[0].reshape(-1)[features[0].reshape(-1) > -10]
    hist = plt.hist(temp, bins=bins, rwidth=5, density=False,
                    alpha=0.5, histtype='stepfilled', color=[0.2656, 0.5625, 0.7656], edgecolor='none')
    plt.plot((hist[1][:-1] + hist[1][1:]) / 2, hist[0], '--', color=[1, 0, 0], alpha=0.5)
    threshold = np.sort(temp)[int(0.05*temp.shape[0])]
    patch_xy = hist[2][0].get_xy()[hist[2][0].get_xy()[:, 0] < threshold]
    patch_xy = np.concatenate([patch_xy[:patch_xy[:, 0].argmax()+2],
                               np.array([[threshold, patch_xy[patch_xy[:, 0].argmax()+1, 1]],
                                         [threshold, 0.]]), patch_xy[patch_xy[:, 0].argmax()+2:]])
    axes.add_patch(PathPatch(Path(patch_xy), alpha=0.8, color=[0.9258, 0.4883, 0.1914]))
    plt.xlabel('Peak amplitude value', fontsize=14)
    plt.ylabel('Numbers of peak', fontsize=14)
    plt.title("Distribution of peak amplitudes\n5th percentile value: " + str(threshold)[:4],
              weight='bold', fontsize=16)
    axes = plt.subplot(132)
    temp = features[1].reshape(-1)[features[1].reshape(-1) > -10]
    hist = plt.hist(temp, bins=bins, rwidth=5, density=False,
                    alpha=0.5, histtype='stepfilled', color='steelblue', edgecolor='none')
    plt.plot((hist[1][:-1] + hist[1][1:]) / 2, hist[0], '--', color=[1, 0, 0], alpha=0.5)
    threshold = np.sort(temp)[int(0.05*temp.shape[0])]
    patch_xy = hist[2][0].get_xy()[hist[2][0].get_xy()[:, 0] < threshold]
    patch_xy = np.concatenate([patch_xy[:patch_xy[:, 0].argmax()+2],
                               np.array([[threshold, patch_xy[patch_xy[:, 0].argmax()+1, 1]],
                                         [threshold, 0.]]), patch_xy[patch_xy[:, 0].argmax()+2:]])
    axes.add_patch(PathPatch(Path(patch_xy), alpha=0.8, color=[0.9258, 0.4883, 0.1914]))
    plt.xlabel('Peak slope value', fontsize=14)
    plt.ylabel('Numbers of peak', fontsize=14)
    plt.title("Distribution of Peak Slopes\n5th percentile value: " + str(threshold)[:4], weight='bold', fontsize=16)
    axes = plt.subplot(133)
    temp = features[2].reshape(-1)[features[2].reshape(-1) > 0.614]
    hist = plt.hist(temp, bins=bins, density=False,
                    alpha=0.5, histtype='stepfilled', color='steelblue', edgecolor='none')
    plt.plot((hist[1][:-1] + hist[1][1:]) / 2, hist[0], '--', color=[1, 0, 0], alpha=0.5)
    threshold = np.sort(temp)[int(0.05*temp.shape[0])]
    patch_xy = hist[2][0].get_xy()[hist[2][0].get_xy()[:, 0] < threshold]
    patch_xy = np.concatenate([patch_xy[:patch_xy[:, 0].argmax()+2],
                               np.array([[threshold, patch_xy[patch_xy[:, 0].argmax()+1, 1]],
                                         [threshold, 0.]]), patch_xy[patch_xy[:, 0].argmax()+2:]])
    axes.add_patch(PathPatch(Path(patch_xy), alpha=0.8, color=[0.9258, 0.4883, 0.1914]))
    plt.xlabel('Peak similarity value', fontsize=14)
    plt.ylabel('Numbers of peak', fontsize=14)
    plt.title("Distribution of Peak Similarities\n5th percentile value: " + str(threshold)[:5],
              weight='bold', fontsize=16)
    plt.show()
