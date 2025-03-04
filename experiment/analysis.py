import os
from typing import Any, Dict

import numpy as np
import torch

from torchays.graph import color, default_subplots


class Analysis:
    def __init__(
        self,
        root_dir,
        with_dataset: bool = False,
    ) -> None:
        self.root_dir = root_dir
        self.with_dataset = with_dataset

    def analysis(self) -> None:
        # draw dataset distribution
        self.common()
        # get data
        experiment_dict = {}
        for tag in os.listdir(self.root_dir):
            if tag in ["analysis"]:
                continue
            tag_dir = os.path.join(self.root_dir, tag)
            if not os.path.isdir(tag_dir):
                continue
            tag_dict = {}
            experiment_dir = os.path.join(tag_dir, "experiment")
            for epoch_fold in os.listdir(experiment_dir):
                epoch = float(epoch_fold.split("_")[-1])
                net_reigions_path = os.path.join(experiment_dir, epoch_fold, 'net_regions.pkl')
                if not os.path.isfile(net_reigions_path):
                    continue
                net_reigions = torch.load(net_reigions_path, weights_only=False)
                tag_dict[epoch] = net_reigions
            experiment_dict[tag] = tag_dict
        # save dir
        save_dir = os.path.join(self.root_dir, "analysis")
        os.makedirs(save_dir, exist_ok=True)
        # draw picture
        self.draw(experiment_dict, save_dir)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.analysis(*args, **kwds)

    def common(self):
        funs = []
        if self.with_dataset:
            funs.append(self.draw_dataset)
        for fun in funs:
            fun()

    def draw_dataset(self):
        dataset_path = os.path.join(self.root_dir, 'dataset.pkl')
        dataset = torch.load(dataset_path, weights_only=False)
        save_path = os.path.join(self.root_dir, "distribution.png")
        x, y, n_classes = dataset['data'], dataset['classes'], dataset['n_classes']
        with default_subplots(save_path, 'x1', 'x2', isLegend=False, isGrid=False) as ax:
            for i in range(n_classes):
                ax.scatter(x[y == i, 0], x[y == i, 1], color=color(i))

    def draw(self, experiment_dict: Dict, save_dir: str):
        funs = [
            self.save_region_epoch_tabel,
            self.draw_region_epoch_plot,
            self.draw_region_acc_plot,
            self.draw_epoch_acc_plot,
        ]
        for fun in funs:
            fun(experiment_dict, save_dir)

    def save_region_epoch_tabel(self, experiment_dict: Dict[str, Dict[Any, Any]], save_dir: str):
        savePath = os.path.join(save_dir, "regionEpoch.csv")
        strBuff = ''
        head = None
        for tag, epochDict in experiment_dict.items():
            tag1 = tag.split('-')[-1].replace(',', '-')
            body = [tag1]
            dataList = []
            for epoch, fileDict in epochDict.items():
                data = [epoch, fileDict['regionNum']]
                dataList.append(data)
            dataList = np.array(dataList)
            a = dataList[:, 0]
            index = np.lexsort((a,))
            dataList = dataList[index]
            regionList = list(map(str, dataList[:, 1].astype(np.int16).tolist()))
            body.extend(regionList)
            bodyStr = ','.join(body)
            if head is None:
                epochList = list(map(str, dataList[:, 0].tolist()))
                head = [
                    'model/epoch',
                ]
                head.extend(epochList)
                headStr = ','.join(head)
                strBuff = strBuff + headStr + '\r\n'
            strBuff = strBuff + bodyStr + '\r\n'
        with open(savePath, 'w') as w:
            w.write(strBuff)
            w.close()

    def draw_region_epoch_plot(self, experiment_dict: Dict[str, Dict[Any, Any]], save_dir: str):
        savePath = os.path.join(save_dir, "regionEpoch.png")
        with default_subplots(savePath, 'Epoch', 'Number of Rgions') as ax:
            i = 0
            for tag, epochDict in experiment_dict.items():
                dataList = []
                for epoch, fileDict in epochDict.items():
                    data = [epoch, fileDict['regionNum']]
                    dataList.append(data)
                dataList = np.array(dataList)
                a = dataList[:, 0]
                index = np.lexsort((a,))
                dataList = dataList[index]
                ax.plot(dataList[:, 0], dataList[:, 1], label=tag, color=color(i))
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
                i += 1

    def draw_region_acc_plot(self, experiment_dict: Dict[str, Dict[Any, Any]], save_dir: str):
        savePath = os.path.join(save_dir, "regionAcc.png")
        with default_subplots(savePath, 'Accuracy', 'Number of Rgions') as ax:
            i = 0
            for tag, epochDict in experiment_dict.items():
                dataList = []
                for _, fileDict in epochDict.items():
                    acc = fileDict['accuracy']
                    if isinstance(acc, torch.Tensor):
                        acc = acc.cpu().numpy()
                    data = [acc, fileDict['regionNum']]
                    dataList.append(data)
                dataList = np.array(dataList)
                a = dataList[:, 0]
                index = np.lexsort((a,))
                dataList = dataList[index]
                ax.plot(dataList[:, 0], dataList[:, 1], label=tag, color=color(i))
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
                # ax.set_xlim(0.965, 0.98)
                i += 1

    def draw_epoch_acc_plot(self, experiment_dict: Dict[str, Dict[Any, Any]], save_dir: str):
        savePath = os.path.join(save_dir, "EpochAcc.png")
        with default_subplots(savePath, 'Epoch', 'Accuracy') as ax:
            i = 0
            for tag, epochDict in experiment_dict.items():
                dataList = []
                for epoch, fileDict in epochDict.items():
                    acc = fileDict['accuracy']
                    if isinstance(acc, torch.Tensor):
                        acc = acc.cpu().numpy()
                    data = [epoch, acc]
                    dataList.append(data)
                dataList = np.array(dataList)
                a = dataList[:, 0]
                index = np.lexsort((a,))
                dataList = dataList[index]
                ax.plot(dataList[:, 0], dataList[:, 1], label=tag, color=color(i))
                i += 1
