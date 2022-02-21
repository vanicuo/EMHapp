# -*- coding:utf-8 -*-
# @Time    : 2022/2/21
# @Author  : cuiwei
# @File    : test.py
# @Software: PyCharm
# @Script to:
#   -
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

if __name__ == '__main__':
    save_dir = '/home/cuiwei/Data_Full/DATA_CHECK'
    png_file_dirs = '/home/cuiwei/Data_Full/MEG_ICA'
    meg_figure_files = [os.path.join(png_file_dirs, x, 'ECG_EOG_FIG')
                        for x in os.listdir(png_file_dirs) if os.path.isdir(os.path.join(png_file_dirs, x))]
    for file in meg_figure_files:
        print(file.split(os.path.sep)[-2])
        figures = [[x,
                    [os.path.join(file, x, y) for y in os.listdir(os.path.join(file, x)) if 'ECG_Samples' in y],
                    [os.path.join(file, x, y) for y in os.listdir(os.path.join(file, x)) if 'EOG_Samples' in y]]
                   for x in np.sort(np.array(os.listdir(file))) if os.path.isdir(os.path.join(file, x))]
        figures = [[x[0],
                    mpimg.imread(x[1][0]) if len(x[1]) > 0 else np.ones([800, 800, 4]),
                    mpimg.imread(x[2][0]) if len(x[2]) > 0 else np.ones([800, 800, 4])] for x in figures]
        figures = [[x[0], np.concatenate((x[1], x[2]), axis=1)] for x in figures]
        # 画图
        figure_col, figsize, dpi = 5, (40, 10), 100
        figure_col = min(figure_col, len(figures))
        fig = plt.figure(num=233, figsize=figsize, clear=True, dpi=dpi)
        # 计算每个fif位置
        cls_range = range(len(figures))
        figure_row = np.ceil(len(cls_range) / figure_col).astype('int64')
        figure_col_row = np.concatenate(
            (np.arange(1, figure_row + 1).reshape(-1, 1).repeat(figure_col, axis=-1).reshape(-1, 1),
             np.arange(0, figure_col).reshape(-1, 1).repeat(figure_row, axis=-1).T.reshape(-1, 1)), axis=-1)
        # 画信号
        for i in cls_range:
            ax = plt.Axes(fig, [1 / figure_col * figure_col_row[i][1], 1 - 1 / figure_row * figure_col_row[i][0],
                                0.95 / figure_col, 0.85 / figure_row])
            ax.set_axis_off()
            fig.add_axes(ax)
            img = (figures[i][1] * np.array([[[0.2989, 0.5870, 0.1140, 0]]])).sum(axis=-1)
            img[0, 0] = 0
            ax.imshow(img, cmap='gray')
            ax.set_title(figures[i][0] + '\n' + 'ECG                                      EOG',
                         fontdict=dict(fontsize=min(18, 36 / figure_row), weight='bold'))
        # 画类别之间的虚线
        for i in range(figure_col_row[:, 0].max() - 1):
            # Plot Line
            ax = plt.Axes(fig, [-1, 1 - 1 / figure_row * np.unique(figure_col_row[:, 0])[i + 1] + 0.95 / figure_row,
                                3, 0.15 / figure_row])
            fig.add_axes(ax)
            ax.plot([0, 1], [0, 0], linestyle='--', color='k')
            ax.set_axis_off()
        for i in range(figure_col_row[:, 1].max()):
            # Plot Line
            ax = plt.Axes(fig, [1 / figure_col * np.unique(figure_col_row[:, 1])[i] + 0.95 / figure_col, -1,
                                0.05 / figure_col, 3])
            fig.add_axes(ax)
            ax.plot([0, 0], [0, 1], linestyle='--', color='k')
            ax.set_axis_off()
        fig.savefig(os.path.join(save_dir, file.split(os.path.sep)[-2] + '.png'))
        plt.close(fig)
