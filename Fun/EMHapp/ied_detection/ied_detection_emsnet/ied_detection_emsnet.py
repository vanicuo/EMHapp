# -*- coding:utf-8 -*-
# @Time    : 2022/2/4
# @Author  : cuiwei
# @File    : ied_detection_emsnet.py
# @Software: PyCharm
# @Script to:
#   - 使用EMG-NET检测ied，模型由Lily提供

import os
import mne
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from ied_detection.ied_detection_emsnet import metrics
import tensorflow as tf
from keras.models import load_model
import keras


class ied_detection_emsnet:
    def __init__(self, raw_data=None, data_info=None, bad_segment=None, device=-1, n_jobs=10):
        """
        Description:
            初始化，使用GPU或者CPU。

        Input:
            :param raw_data: ndarray, double, shape(channel, samples)
                MEG滤波后数据
            :param data_info: dict
                MEG数据的信息, MNE读取的raw.info
            :param bad_segment: ndarray, bool, shape(1, samples)
                MEG数据中的坏片段，True代表对应时间点属于坏片段
            :param device: number, int
                device<0 使用CPU, device>=0 使用对应GPU
            :param n_jobs: number, int
                MEN函数中，使用到的并行个数
        """
        # 使用GPU或者CPU
        self.device_number = device
        self.n_jobs = n_jobs
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self._check_cuda(device)
        # 变量初始化
        self.channels_in_roi = {
            1: [[
                'MEG0223', 'MEG0222', 'MEG0212', 'MEG0213', 'MEG0133',
                'MEG0132', 'MEG0112', 'MEG0113', 'MEG0233', 'MEG0232',
                'MEG0243', 'MEG0242', 'MEG1512', 'MEG1513', 'MEG0143',
                'MEG0142', 'MEG1623', 'MEG1622', 'MEG1613', 'MEG1612',
                'MEG1523', 'MEG1522', 'MEG1543', 'MEG1542', 'MEG1533',
                'MEG1532', 'MEG0221', 'MEG0211', 'MEG0131', 'MEG0111',
                'MEG0231', 'MEG0241', 'MEG1511', 'MEG0141', 'MEG1621',
                'MEG1611', 'MEG1521', 'MEG1541', 'MEG1531'
            ]],
            2: [[
                'MEG1312', 'MEG1313', 'MEG1323', 'MEG1322', 'MEG1442',
                'MEG1443', 'MEG1423', 'MEG1422', 'MEG1342', 'MEG1343',
                'MEG1333', 'MEG1332', 'MEG2612', 'MEG2613', 'MEG1433',
                'MEG1432', 'MEG2413', 'MEG2412', 'MEG2422', 'MEG2423',
                'MEG2642', 'MEG2643', 'MEG2623', 'MEG2622', 'MEG2633',
                'MEG2632', 'MEG1311', 'MEG1321', 'MEG1441', 'MEG1421',
                'MEG1341', 'MEG1331', 'MEG2611', 'MEG1431', 'MEG2411',
                'MEG2421', 'MEG2641', 'MEG2621', 'MEG2631'
            ]],
            3: [[
                'MEG0633', 'MEG0632', 'MEG0423', 'MEG0422', 'MEG0412',
                'MEG0413', 'MEG0712', 'MEG0713', 'MEG0433', 'MEG0432',
                'MEG0442', 'MEG0443', 'MEG0742', 'MEG0743', 'MEG1822',
                'MEG1823', 'MEG1813', 'MEG1812', 'MEG1832', 'MEG1833',
                'MEG1843', 'MEG1842', 'MEG1632', 'MEG1633', 'MEG2013',
                'MEG2012', 'MEG0631', 'MEG0421', 'MEG0411', 'MEG0711',
                'MEG0431', 'MEG0441', 'MEG0741', 'MEG1821', 'MEG1811',
                'MEG1831', 'MEG1841', 'MEG1631', 'MEG2011'
            ]],
            4: [[
                'MEG1043', 'MEG1042', 'MEG1112', 'MEG1113', 'MEG1123',
                'MEG1122', 'MEG0722', 'MEG0723', 'MEG1142', 'MEG1143',
                'MEG1133', 'MEG1132', 'MEG0732', 'MEG0733', 'MEG2212',
                'MEG2213', 'MEG2223', 'MEG2222', 'MEG2242', 'MEG2243',
                'MEG2232', 'MEG2233', 'MEG2442', 'MEG2443', 'MEG2023',
                'MEG2022', 'MEG1041', 'MEG1111', 'MEG1121', 'MEG0721',
                'MEG1141', 'MEG1131', 'MEG0731', 'MEG2211', 'MEG2221',
                'MEG2241', 'MEG2231', 'MEG2441', 'MEG2021'
            ]],
            5: [[
                'MEG2042', 'MEG2043', 'MEG1913', 'MEG1912', 'MEG2113',
                'MEG2112', 'MEG1922', 'MEG1923', 'MEG1942', 'MEG1943',
                'MEG1642', 'MEG1643', 'MEG1933', 'MEG1932', 'MEG1733',
                'MEG1732', 'MEG1723', 'MEG1722', 'MEG2143', 'MEG2142',
                'MEG1742', 'MEG1743', 'MEG1712', 'MEG1713', 'MEG2041',
                'MEG1911', 'MEG2111', 'MEG1921', 'MEG1941', 'MEG1641',
                'MEG1931', 'MEG1731', 'MEG1721', 'MEG2141', 'MEG1741',
                'MEG1711'
            ]],
            6: [[
                'MEG2032', 'MEG2033', 'MEG2313', 'MEG2312', 'MEG2342',
                'MEG2343', 'MEG2322', 'MEG2323', 'MEG2433', 'MEG2432',
                'MEG2122', 'MEG2123', 'MEG2333', 'MEG2332', 'MEG2513',
                'MEG2512', 'MEG2523', 'MEG2522', 'MEG2133', 'MEG2132',
                'MEG2542', 'MEG2543', 'MEG2532', 'MEG2533', 'MEG2031',
                'MEG2311', 'MEG2341', 'MEG2321', 'MEG2431', 'MEG2121',
                'MEG2331', 'MEG2511', 'MEG2521', 'MEG2131', 'MEG2541',
                'MEG2531'
            ]],
            7: [[
                'MEG0522', 'MEG0523', 'MEG0512', 'MEG0513', 'MEG0312',
                'MEG0313', 'MEG0342', 'MEG0343', 'MEG0122', 'MEG0123',
                'MEG0822', 'MEG0823', 'MEG0533', 'MEG0532', 'MEG0543',
                'MEG0542', 'MEG0322', 'MEG0323', 'MEG0612', 'MEG0613',
                'MEG0333', 'MEG0332', 'MEG0622', 'MEG0623', 'MEG0643',
                'MEG0642', 'MEG0521', 'MEG0511', 'MEG0311', 'MEG0341',
                'MEG0121', 'MEG0821', 'MEG0531', 'MEG0541', 'MEG0321',
                'MEG0611', 'MEG0331', 'MEG0621', 'MEG0641'
            ]],
            8: [[
                'MEG0813', 'MEG0812', 'MEG0912', 'MEG0913', 'MEG0922',
                'MEG0923', 'MEG1212', 'MEG1213', 'MEG1223', 'MEG1222',
                'MEG1412', 'MEG1413', 'MEG0943', 'MEG0942', 'MEG0933',
                'MEG0932', 'MEG1232', 'MEG1233', 'MEG1012', 'MEG1013',
                'MEG1022', 'MEG1023', 'MEG1243', 'MEG1242', 'MEG1033',
                'MEG1032', 'MEG0811', 'MEG0911', 'MEG0921', 'MEG1211',
                'MEG1221', 'MEG1411', 'MEG0941', 'MEG0931', 'MEG1231',
                'MEG1011', 'MEG1021', 'MEG1241', 'MEG1031'
            ]],
            9: [[
                'MEG0223', 'MEG0222', 'MEG0212', 'MEG0213', 'MEG0133',
                'MEG0132', 'MEG0112', 'MEG0113', 'MEG0233', 'MEG0232',
                'MEG0243', 'MEG0242', 'MEG1512', 'MEG1513', 'MEG0143',
                'MEG0142', 'MEG1623', 'MEG1622', 'MEG1613', 'MEG1612',
                'MEG1523', 'MEG1522', 'MEG1543', 'MEG1542', 'MEG1533',
                'MEG1532', 'MEG0221', 'MEG0211', 'MEG0131', 'MEG0111',
                'MEG0231', 'MEG0241', 'MEG1511', 'MEG0141', 'MEG1621',
                'MEG1611', 'MEG1521', 'MEG1541', 'MEG1531'
            ],
                [
                    'MEG0633', 'MEG0632', 'MEG0423', 'MEG0422', 'MEG0412',
                    'MEG0413', 'MEG0712', 'MEG0713', 'MEG0433', 'MEG0432',
                    'MEG0442', 'MEG0443', 'MEG0742', 'MEG0743', 'MEG1822',
                    'MEG1823', 'MEG1813', 'MEG1812', 'MEG1832', 'MEG1833',
                    'MEG1843', 'MEG1842', 'MEG1632', 'MEG1633', 'MEG2013',
                    'MEG2012', 'MEG0631', 'MEG0421', 'MEG0411', 'MEG0711',
                    'MEG0431', 'MEG0441', 'MEG0741', 'MEG1821', 'MEG1811',
                    'MEG1831', 'MEG1841', 'MEG1631', 'MEG2011'
                ],
                [
                    'MEG2042', 'MEG2043', 'MEG1913', 'MEG1912', 'MEG2113',
                    'MEG2112', 'MEG1922', 'MEG1923', 'MEG1942', 'MEG1943',
                    'MEG1642', 'MEG1643', 'MEG1933', 'MEG1932', 'MEG1733',
                    'MEG1732', 'MEG1723', 'MEG1722', 'MEG2143', 'MEG2142',
                    'MEG1742', 'MEG1743', 'MEG1712', 'MEG1713', 'MEG2041',
                    'MEG1911', 'MEG2111', 'MEG1921', 'MEG1941', 'MEG1641',
                    'MEG1931', 'MEG1731', 'MEG1721', 'MEG2141', 'MEG1741',
                    'MEG1711'
                ],
                [
                    'MEG0522', 'MEG0523', 'MEG0512', 'MEG0513', 'MEG0312',
                    'MEG0313', 'MEG0342', 'MEG0343', 'MEG0122', 'MEG0123',
                    'MEG0822', 'MEG0823', 'MEG0533', 'MEG0532', 'MEG0543',
                    'MEG0542', 'MEG0322', 'MEG0323', 'MEG0612', 'MEG0613',
                    'MEG0333', 'MEG0332', 'MEG0622', 'MEG0623', 'MEG0643',
                    'MEG0642', 'MEG0521', 'MEG0511', 'MEG0311', 'MEG0341',
                    'MEG0121', 'MEG0821', 'MEG0531', 'MEG0541', 'MEG0321',
                    'MEG0611', 'MEG0331', 'MEG0621', 'MEG0641'
                ]],
            10: [[
                'MEG1312', 'MEG1313', 'MEG1323', 'MEG1322', 'MEG1442',
                'MEG1443', 'MEG1423', 'MEG1422', 'MEG1342', 'MEG1343',
                'MEG1333', 'MEG1332', 'MEG2612', 'MEG2613', 'MEG1433',
                'MEG1432', 'MEG2413', 'MEG2412', 'MEG2422', 'MEG2423',
                'MEG2642', 'MEG2643', 'MEG2623', 'MEG2622', 'MEG2633',
                'MEG2632', 'MEG1311', 'MEG1321', 'MEG1441', 'MEG1421',
                'MEG1341', 'MEG1331', 'MEG2611', 'MEG1431', 'MEG2411',
                'MEG2421', 'MEG2641', 'MEG2621', 'MEG2631'
            ],
                [
                    'MEG1043', 'MEG1042', 'MEG1112', 'MEG1113', 'MEG1123',
                    'MEG1122', 'MEG0722', 'MEG0723', 'MEG1142', 'MEG1143',
                    'MEG1133', 'MEG1132', 'MEG0732', 'MEG0733', 'MEG2212',
                    'MEG2213', 'MEG2223', 'MEG2222', 'MEG2242', 'MEG2243',
                    'MEG2232', 'MEG2233', 'MEG2442', 'MEG2443', 'MEG2023',
                    'MEG2022', 'MEG1041', 'MEG1111', 'MEG1121', 'MEG0721',
                    'MEG1141', 'MEG1131', 'MEG0731', 'MEG2211', 'MEG2221',
                    'MEG2241', 'MEG2231', 'MEG2441', 'MEG2021'
                ],
                [
                    'MEG2032', 'MEG2033', 'MEG2313', 'MEG2312', 'MEG2342',
                    'MEG2343', 'MEG2322', 'MEG2323', 'MEG2433', 'MEG2432',
                    'MEG2122', 'MEG2123', 'MEG2333', 'MEG2332', 'MEG2513',
                    'MEG2512', 'MEG2523', 'MEG2522', 'MEG2133', 'MEG2132',
                    'MEG2542', 'MEG2543', 'MEG2532', 'MEG2533', 'MEG2031',
                    'MEG2311', 'MEG2341', 'MEG2321', 'MEG2431', 'MEG2121',
                    'MEG2331', 'MEG2511', 'MEG2521', 'MEG2131', 'MEG2541',
                    'MEG2531'
                ],
                [
                    'MEG0813', 'MEG0812', 'MEG0912', 'MEG0913', 'MEG0922',
                    'MEG0923', 'MEG1212', 'MEG1213', 'MEG1223', 'MEG1222',
                    'MEG1412', 'MEG1413', 'MEG0943', 'MEG0942', 'MEG0933',
                    'MEG0932', 'MEG1232', 'MEG1233', 'MEG1012', 'MEG1013',
                    'MEG1022', 'MEG1023', 'MEG1243', 'MEG1242', 'MEG1033',
                    'MEG1032', 'MEG0811', 'MEG0911', 'MEG0921', 'MEG1211',
                    'MEG1221', 'MEG1411', 'MEG0941', 'MEG0931', 'MEG1231',
                    'MEG1011', 'MEG1021', 'MEG1241', 'MEG1031'
                ]],
            100: [[
                'MEG0223', 'MEG0222', 'MEG0212', 'MEG0213', 'MEG0133',
                'MEG0132', 'MEG0112', 'MEG0113', 'MEG0233', 'MEG0232',
                'MEG0243', 'MEG0242', 'MEG1512', 'MEG1513', 'MEG0143',
                'MEG0142', 'MEG1623', 'MEG1622', 'MEG1613', 'MEG1612',
                'MEG1523', 'MEG1522', 'MEG1543', 'MEG1542', 'MEG1533',
                'MEG1532', 'MEG0221', 'MEG0211', 'MEG0131', 'MEG0111',
                'MEG0231', 'MEG0241', 'MEG1511', 'MEG0141', 'MEG1621',
                'MEG1611', 'MEG1521', 'MEG1541', 'MEG1531'
            ],
                [
                    'MEG1312', 'MEG1313', 'MEG1323', 'MEG1322', 'MEG1442',
                    'MEG1443', 'MEG1423', 'MEG1422', 'MEG1342', 'MEG1343',
                    'MEG1333', 'MEG1332', 'MEG2612', 'MEG2613', 'MEG1433',
                    'MEG1432', 'MEG2413', 'MEG2412', 'MEG2422', 'MEG2423',
                    'MEG2642', 'MEG2643', 'MEG2623', 'MEG2622', 'MEG2633',
                    'MEG2632', 'MEG1311', 'MEG1321', 'MEG1441', 'MEG1421',
                    'MEG1341', 'MEG1331', 'MEG2611', 'MEG1431', 'MEG2411',
                    'MEG2421', 'MEG2641', 'MEG2621', 'MEG2631'
                ],
                [
                    'MEG0633', 'MEG0632', 'MEG0423', 'MEG0422', 'MEG0412',
                    'MEG0413', 'MEG0712', 'MEG0713', 'MEG0433', 'MEG0432',
                    'MEG0442', 'MEG0443', 'MEG0742', 'MEG0743', 'MEG1822',
                    'MEG1823', 'MEG1813', 'MEG1812', 'MEG1832', 'MEG1833',
                    'MEG1843', 'MEG1842', 'MEG1632', 'MEG1633', 'MEG2013',
                    'MEG2012', 'MEG0631', 'MEG0421', 'MEG0411', 'MEG0711',
                    'MEG0431', 'MEG0441', 'MEG0741', 'MEG1821', 'MEG1811',
                    'MEG1831', 'MEG1841', 'MEG1631', 'MEG2011'
                ],
                [
                    'MEG1043', 'MEG1042', 'MEG1112', 'MEG1113', 'MEG1123',
                    'MEG1122', 'MEG0722', 'MEG0723', 'MEG1142', 'MEG1143',
                    'MEG1133', 'MEG1132', 'MEG0732', 'MEG0733', 'MEG2212',
                    'MEG2213', 'MEG2223', 'MEG2222', 'MEG2242', 'MEG2243',
                    'MEG2232', 'MEG2233', 'MEG2442', 'MEG2443', 'MEG2023',
                    'MEG2022', 'MEG1041', 'MEG1111', 'MEG1121', 'MEG0721',
                    'MEG1141', 'MEG1131', 'MEG0731', 'MEG2211', 'MEG2221',
                    'MEG2241', 'MEG2231', 'MEG2441', 'MEG2021'
                ],
                [
                    'MEG2042', 'MEG2043', 'MEG1913', 'MEG1912', 'MEG2113',
                    'MEG2112', 'MEG1922', 'MEG1923', 'MEG1942', 'MEG1943',
                    'MEG1642', 'MEG1643', 'MEG1933', 'MEG1932', 'MEG1733',
                    'MEG1732', 'MEG1723', 'MEG1722', 'MEG2143', 'MEG2142',
                    'MEG1742', 'MEG1743', 'MEG1712', 'MEG1713', 'MEG2041',
                    'MEG1911', 'MEG2111', 'MEG1921', 'MEG1941', 'MEG1641',
                    'MEG1931', 'MEG1731', 'MEG1721', 'MEG2141', 'MEG1741',
                    'MEG1711'
                ],
                [
                    'MEG2032', 'MEG2033', 'MEG2313', 'MEG2312', 'MEG2342',
                    'MEG2343', 'MEG2322', 'MEG2323', 'MEG2433', 'MEG2432',
                    'MEG2122', 'MEG2123', 'MEG2333', 'MEG2332', 'MEG2513',
                    'MEG2512', 'MEG2523', 'MEG2522', 'MEG2133', 'MEG2132',
                    'MEG2542', 'MEG2543', 'MEG2532', 'MEG2533', 'MEG2031',
                    'MEG2311', 'MEG2341', 'MEG2321', 'MEG2431', 'MEG2121',
                    'MEG2331', 'MEG2511', 'MEG2521', 'MEG2131', 'MEG2541',
                    'MEG2531'
                ],
                [
                    'MEG0522', 'MEG0523', 'MEG0512', 'MEG0513', 'MEG0312',
                    'MEG0313', 'MEG0342', 'MEG0343', 'MEG0122', 'MEG0123',
                    'MEG0822', 'MEG0823', 'MEG0533', 'MEG0532', 'MEG0543',
                    'MEG0542', 'MEG0322', 'MEG0323', 'MEG0612', 'MEG0613',
                    'MEG0333', 'MEG0332', 'MEG0622', 'MEG0623', 'MEG0643',
                    'MEG0642', 'MEG0521', 'MEG0511', 'MEG0311', 'MEG0341',
                    'MEG0121', 'MEG0821', 'MEG0531', 'MEG0541', 'MEG0321',
                    'MEG0611', 'MEG0331', 'MEG0621', 'MEG0641'
                ],
                [
                    'MEG0813', 'MEG0812', 'MEG0912', 'MEG0913', 'MEG0922',
                    'MEG0923', 'MEG1212', 'MEG1213', 'MEG1223', 'MEG1222',
                    'MEG1412', 'MEG1413', 'MEG0943', 'MEG0942', 'MEG0933',
                    'MEG0932', 'MEG1232', 'MEG1233', 'MEG1012', 'MEG1013',
                    'MEG1022', 'MEG1023', 'MEG1243', 'MEG1242', 'MEG1033',
                    'MEG1032', 'MEG0811', 'MEG0911', 'MEG0921', 'MEG1211',
                    'MEG1221', 'MEG1411', 'MEG0941', 'MEG0931', 'MEG1231',
                    'MEG1011', 'MEG1021', 'MEG1241', 'MEG1031'
                ]],
            101: [
                ['MEG0223', 'MEG0222', 'MEG0212', 'MEG0213', 'MEG0133', 'MEG0132', 'MEG0112', 'MEG0113', 'MEG0233',
                 'MEG0232', 'MEG0243', 'MEG0242', 'MEG1512', 'MEG1513', 'MEG0143', 'MEG0142', 'MEG1623', 'MEG1622',
                 'MEG1613', 'MEG1612', 'MEG1523', 'MEG1522', 'MEG1543', 'MEG1542', 'MEG1533', 'MEG1532', 'MEG0221',
                 'MEG0211', 'MEG0131', 'MEG0111', 'MEG0231', 'MEG0241', 'MEG1511', 'MEG0141', 'MEG1621', 'MEG1611',
                 'MEG1521', 'MEG1541', 'MEG1531', 'MEG1312', 'MEG1313', 'MEG1323', 'MEG1322', 'MEG1442', 'MEG1443',
                 'MEG1423', 'MEG1422', 'MEG1342', 'MEG1343', 'MEG1333', 'MEG1332', 'MEG2612', 'MEG2613', 'MEG1433',
                 'MEG1432', 'MEG2413', 'MEG2412', 'MEG2422', 'MEG2423', 'MEG2642', 'MEG2643', 'MEG2623', 'MEG2622',
                 'MEG2633', 'MEG2632', 'MEG1311', 'MEG1321', 'MEG1441', 'MEG1421', 'MEG1341', 'MEG1331', 'MEG2611',
                 'MEG1431', 'MEG2411', 'MEG2421', 'MEG2641', 'MEG2621', 'MEG2631', 'MEG0633', 'MEG0632', 'MEG0423',
                 'MEG0422', 'MEG0412', 'MEG0413', 'MEG0712', 'MEG0713', 'MEG0433', 'MEG0432', 'MEG0442', 'MEG0443',
                 'MEG0742', 'MEG0743', 'MEG1822', 'MEG1823', 'MEG1813', 'MEG1812', 'MEG1832', 'MEG1833', 'MEG1843',
                 'MEG1842', 'MEG1632', 'MEG1633', 'MEG2013', 'MEG2012', 'MEG0631', 'MEG0421', 'MEG0411', 'MEG0711',
                 'MEG0431', 'MEG0441', 'MEG0741', 'MEG1821', 'MEG1811', 'MEG1831', 'MEG1841', 'MEG1631', 'MEG2011',
                 'MEG1043', 'MEG1042', 'MEG1112', 'MEG1113', 'MEG1123', 'MEG1122', 'MEG0722', 'MEG0723', 'MEG1142',
                 'MEG1143', 'MEG1133', 'MEG1132', 'MEG0732', 'MEG0733', 'MEG2212', 'MEG2213', 'MEG2223', 'MEG2222',
                 'MEG2242', 'MEG2243', 'MEG2232', 'MEG2233', 'MEG2442', 'MEG2443', 'MEG2023', 'MEG2022', 'MEG1041',
                 'MEG1111', 'MEG1121', 'MEG0721', 'MEG1141', 'MEG1131', 'MEG0731', 'MEG2211', 'MEG2221', 'MEG2241',
                 'MEG2231', 'MEG2441', 'MEG2021', 'MEG2042', 'MEG2043', 'MEG1913', 'MEG1912', 'MEG2113', 'MEG2112',
                 'MEG1922', 'MEG1923', 'MEG1942', 'MEG1943', 'MEG1642', 'MEG1643', 'MEG1933', 'MEG1932', 'MEG1733',
                 'MEG1732', 'MEG1723', 'MEG1722', 'MEG2143', 'MEG2142', 'MEG1742', 'MEG1743', 'MEG1712', 'MEG1713',
                 'MEG2041', 'MEG1911', 'MEG2111', 'MEG1921', 'MEG1941', 'MEG1641', 'MEG1931', 'MEG1731', 'MEG1721',
                 'MEG2141', 'MEG1741', 'MEG1711', 'MEG2032', 'MEG2033', 'MEG2313', 'MEG2312', 'MEG2342', 'MEG2343',
                 'MEG2322', 'MEG2323', 'MEG2433', 'MEG2432', 'MEG2122', 'MEG2123', 'MEG2333', 'MEG2332', 'MEG2513',
                 'MEG2512', 'MEG2523', 'MEG2522', 'MEG2133', 'MEG2132', 'MEG2542', 'MEG2543', 'MEG2532', 'MEG2533',
                 'MEG2031', 'MEG2311', 'MEG2341', 'MEG2321', 'MEG2431', 'MEG2121', 'MEG2331', 'MEG2511', 'MEG2521',
                 'MEG2131', 'MEG2541', 'MEG2531', 'MEG0522', 'MEG0523', 'MEG0512', 'MEG0513', 'MEG0312', 'MEG0313',
                 'MEG0342', 'MEG0343', 'MEG0122', 'MEG0123', 'MEG0822', 'MEG0823', 'MEG0533', 'MEG0532', 'MEG0543',
                 'MEG0542', 'MEG0322', 'MEG0323', 'MEG0612', 'MEG0613', 'MEG0333', 'MEG0332', 'MEG0622', 'MEG0623',
                 'MEG0643', 'MEG0642', 'MEG0521', 'MEG0511', 'MEG0311', 'MEG0341', 'MEG0121', 'MEG0821', 'MEG0531',
                 'MEG0541', 'MEG0321', 'MEG0611', 'MEG0331', 'MEG0621', 'MEG0641', 'MEG0813', 'MEG0812', 'MEG0912',
                 'MEG0913', 'MEG0922', 'MEG0923', 'MEG1212', 'MEG1213', 'MEG1223', 'MEG1222', 'MEG1412', 'MEG1413',
                 'MEG0943', 'MEG0942', 'MEG0933', 'MEG0932', 'MEG1232', 'MEG1233', 'MEG1012', 'MEG1013', 'MEG1022',
                 'MEG1023', 'MEG1243', 'MEG1242', 'MEG1033', 'MEG1032', 'MEG0811', 'MEG0911', 'MEG0921', 'MEG1211',
                 'MEG1221', 'MEG1411', 'MEG0941', 'MEG0931', 'MEG1231', 'MEG1011', 'MEG1021', 'MEG1241', 'MEG1031']
            ]

        }
        self.raw_data, self.data_info, self.bad_segment = raw_data, data_info, bad_segment
        self.set_raw_data(raw_data)
        self.set_data_info(data_info)
        self.set_bad_segment(bad_segment)

    def _check_cuda(self, device):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        if device > -1:
            # Init MEN cuda
            try:
                mne.cuda.set_cuda_device(device, verbose=False)
                mne.utils.set_config('MNE_USE_CUDA', 'true', verbose=False)
                mne.cuda.init_cuda(verbose=False)
                self.is_cuda = 1
            except:
                self.is_cuda = 0
            # Init torch cuda
            if torch.cuda.is_available():
                # Init tensorflow
                gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                # Init torch
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
            self.is_cuda = 0

    def del_var(self, *arg):
        """
        Description:
            清除变量，释放缓存

        Input:
            :param arg: list, string
                需要清除的变量名称
        """
        if arg is not None:
            for key in list(globals().keys()):
                if key in arg:
                    globals()[key] = []
        if self.device == torch.device("cuda"):
            torch.cuda.empty_cache()

    def set_raw_data(self, raw_data=None):
        assert raw_data is not None
        self.raw_data = torch.tensor(raw_data)

    def get_raw_data(self):
        return self.raw_data

    def set_data_info(self, data_info=None):
        assert data_info is not None
        self.data_info = data_info

    def get_data_info(self):
        return self.data_info

    def set_bad_segment(self, bad_segment=None):
        if bad_segment is None:
            bad_segment = np.ones(self.get_raw_data().shape[1]) < 0
        self.bad_segment = torch.tensor(bad_segment)

    def get_bad_segment(self):
        return self.bad_segment

    def cal_segment_sample_index(self, segment_windows=0.3, segment_overlap=0.5):
        """
        Description:
            根据片段长度和片段之间的重合度，确定每个片段在原始数据中的sample index。并去除包含bad segment的片段。

        Input:
            :param segment_windows: number, double
                每个片段的长度，单位S
            :param segment_overlap: number, double
                相邻片段之间的重合度
        Return:
            :return segment_samples: torch.tensor, double, shape(n_segments*n_segment_samples)
                每个片段在原始数据中的sample index
        """

        with torch.no_grad():
            segment_windows = int(segment_windows * self.get_data_info()['sfreq'])
            # 获取meg数据和坏片段
            meg_data = self.get_raw_data().clone().to(self.device)
            bad_segments = self.get_bad_segment().clone().to(self.device)

            # 根据segment长度和segment之间的重合度，确定每个segment的sample index
            segment_samples = (torch.arange(0, meg_data.shape[1], int(segment_windows * segment_overlap)).unsqueeze(1) +
                               torch.arange(0, segment_windows).unsqueeze(0))[:-1]
            # 确保segment_samples在数据范围内
            segment_samples = segment_samples[torch.where(segment_samples[:, -1] < meg_data.shape[-1])[0]]
            # 去除包含bad segment的segment
            bad_segment_samples = bad_segments[segment_samples]
            segment_samples = segment_samples[~bad_segment_samples.any(dim=-1)]
        self.del_var()

        return segment_samples

    def cal_roi_channel_index(self):
        """
        Description:
            获取每个roi内对应的channel index

        Return:
            :return roi_channel_index: list, [torch.tensor, long, shape(n_channels)], shape(n_rois)
                每个roi内对应的channel index
        """

        channel_name = [x[0] for i, x in enumerate(self.channels_in_roi.values()) if i < 8]
        roi_channel_index = [torch.tensor([self.get_data_info().ch_names.index(y)
                                           for y in x if y in self.get_data_info().ch_names]) for x in channel_name]

        return roi_channel_index

    def cut_data_to_segments(self, segment_samples, roi_channel_index):
        """
        Description:
            根据segment_samples和roi_channel_index，对数据进行切片

        Input:
            :param segment_samples: torch.tensor, double, shape(n_segments*n_segment_samples)
                每个片段在原始数据中的sample index
            :param roi_channel_index: list, [torch.tensor, long, shape(n_channels)], shape(n_rois)
                每个roi内对应的channel index
        Return:
            :return roi_segment_data: list, [torch.tensor, double, shape(n_segments*39*n_segment_samples)], shape(n_rois)
                切片后的数据片段
        """

        with torch.no_grad():
            # 获取meg数据和坏片段
            meg_data = self.get_raw_data().clone().to(self.device)

            # 将数据切片成片段
            meg_data = meg_data[:, segment_samples].permute(1, 0, 2)

            # 将片段数据切片成ROI
            roi_segment_data = []
            for index in roi_channel_index:
                if index.shape[0] == 36:
                    # 如果index的长度小于39，则补零
                    temp = meg_data[:, index]
                    roi_segment_data_temp = torch.cat((temp[:, :24], torch.zeros_like(temp)[:, :2],
                                                       temp[:, 24:], torch.zeros_like(temp)[:, :1]),
                                                      dim=1)
                else:
                    # 如果index的长度等于39，则直接切片
                    roi_segment_data_temp = meg_data[:, index]
                roi_segment_data.append(roi_segment_data_temp)
            roi_segment_data = torch.stack(roi_segment_data)
        self.del_var()

        return roi_segment_data

    @staticmethod
    def data_normalization(roi_segment_data, grad_index=range(0, 26), mag_index=range(26, 39)):
        """
        Description:
            对切片后的数据进行归一化，使用ROI内全局归一化。

        Input:
            :param roi_segment_data: list, [torch.tensor, double, shape(n_segments*39*n_segment_samples)], shape(n_rois)
                切片后的数据片段
            :param grad_index: list, long, shape(n_grad_channels)
                每个ROI内，grad channel的index
            :param mag_index: list, long, shape(n_mag_channels)
                每个ROI内，mag channel的index
        Return:
            :return normalized_roi_segment_data: list, [np.array, double, shape(n_segments*39*n_segment_samples)], shape(n_rois)
                归一化切片后的数据
        """
        with torch.no_grad():
            normalized_roi_segment_data = []
            for data in roi_segment_data:
                nor_data = torch.zeros_like(data)
                # 去除补零的通道
                none_zeros_index = (data != 0).sum(dim=(0, 2)) != 0
                grad_idx = torch.tensor(grad_index)[none_zeros_index[grad_index]]
                mag_idx = torch.tensor(mag_index)[none_zeros_index[mag_index]]
                # 进行归一化
                nor_data[:, grad_idx] = \
                    (data[:, grad_idx] - torch.mean(data[:, grad_idx])) / torch.std(data[:, grad_idx])
                nor_data[:, mag_idx] = \
                    (data[:, mag_idx] - torch.mean(data[:, mag_idx])) / torch.std(data[:, mag_idx])
                normalized_roi_segment_data.append(nor_data.cpu().numpy())

        return normalized_roi_segment_data

    def ied_detection_emsnet(self, segment_windows=0.3, segment_overlap=0.5, emsnet_threshold=0.9, model_path=None):
        """
        Description:
            使用ems net，检测MEG中的IED片段

        Input:
            :param segment_windows: number, double
                每个片段的长度，单位S
            :param segment_overlap: number, double
                相邻片段之间的重合度
            :param emsnet_threshold: number, double
                判断为ied的阈值
            :param model_path: path
                ems net的模型路径
        Return:
            :return candidate_ied_win: np.array, long, shape(n_ieds*n_segment_samples)
                每个ied时间窗在原始MEG数据中的采样点位置
        """

        # step1: 加载模型
        ems_net = keras.models.load_model(
            custom_objects={'recall': metrics.recall, 'recall_0.3': metrics.recall_threshold(0.3),
                            'precision': metrics.precision, 'precision_0.3': metrics.precision_threshold(0.3),
                            'f1_score': metrics.f1_score, 'f1_score_0.3': metrics.f1_score_threshold(0.3)},
            filepath=model_path)

        # step2: 切数据
        # 获得片段的sample index
        segment_samples = self.cal_segment_sample_index(segment_windows=segment_windows,
                                                        segment_overlap=segment_overlap)
        self.del_var()
        # 获得roi的channel index
        roi_channel_index = self.cal_roi_channel_index()
        # 切数据
        roi_segment_data = self.cut_data_to_segments(segment_samples=segment_samples,
                                                     roi_channel_index=roi_channel_index)
        self.del_var()

        # step3: 对数据，进行归一化
        normalized_roi_segment_data = self.data_normalization(roi_segment_data=roi_segment_data)
        del roi_segment_data
        self.del_var()

        # step4: 使用模型进行预测
        predict_results = []
        for data in normalized_roi_segment_data:
            data = np.expand_dims(data, axis=3)
            prediction_keras = ems_net.predict(data)
            predict_results.append(prediction_keras)
        predict_results = np.concatenate(predict_results, axis=-1).max(axis=-1)
        candidate_ied_win = segment_samples[np.where(predict_results > emsnet_threshold)[0]].cpu().numpy()
        candidate_ied_win = candidate_ied_win if len(candidate_ied_win.shape) == 2 \
            else candidate_ied_win.reshape(1, candidate_ied_win.shape[0])

        del normalized_roi_segment_data
        self.del_var()

        return candidate_ied_win


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(2)

    raw = mne.io.read_raw_fif('/sanbo_dataset/4k_Project/Data_Full/MEG_IED/MEG1781/MEG1781_EP_1_tsss.fif',
                              verbose='error', preload=True)
    raw.pick(mne.pick_types(raw.info, meg=True, ref_meg=False))
    Info = raw.info
    Raw_data = raw.get_data()

    # 计算发作间期IEDs
    IED = ied_detection_emsnet(raw_data=Raw_data, data_info=Info, device=2)
    IED.ied_detection_emsnet(segment_windows=0.3, segment_overlap=0.5, emsnet_threshold=0.8,
                             model_path='/data2/cuiwei/HFO_IED_Ictal/EMHapp/ied_detection/ied_detection_emsnet/best_run_4.h5')
