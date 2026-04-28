import csv
import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.contrib import tenumerate
from src.dataset.base_dataset import TrainDataset, EvalDataset
import importlib.util


def make_model(conf):
    model_path = conf['model_path']
    model_name = conf['model_name']
    module = importlib.import_module(f'.model.{model_path}', 'src')
    MyModel = getattr(module, model_name)
    model = MyModel(conf)
    return model


class Trainer(object):
    def __init__(self, conf):
        super().__init__()
        self.mode = conf["mode"]
        self.model_save_dir = conf["model_save_dir"]
        self.global_step = 0

        ######################################## Dataset & Dataloader ########################################
        if self.mode == "train":
            train_dataset = TrainDataset(conf)
            self.train_dataloader = DataLoader(dataset=train_dataset,
                                               batch_size=10,
                                               shuffle=True,
                                               num_workers=4,
                                               pin_memory=True,
                                               drop_last=True)
        self.eval_dataloader_dic = {}
        eval_label_files = conf["eval_label_files"]
        for eval_label_file in eval_label_files:
            eval_dataset = EvalDataset(conf, eval_label_file)
            eval_dataloader = DataLoader(dataset=eval_dataset,
                                         batch_size=10,
                                         shuffle=False,
                                         num_workers=4,
                                         pin_memory=True,
                                         drop_last=False)
            self.eval_dataloader_dic[eval_label_file] = eval_dataloader

        ########################################### Model ###########################################
        self.model = make_model(conf).to(conf['device'])

        #################################### Optimizer #################################################
        if self.mode == "train":
            if conf['optimizer'] == 'adam':
                self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                  lr=0.0001)
            elif conf['optimizer'] == 'sgd':
                self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                 lr=0.0001, momentum=0.9,
                                                 weight_decay=0.0000001)
            else:
                return

        ###################################### Learning Rate Scheduler ###############################################
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 40, gamma=0.45)

    def reload(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        model_state_load = checkpoint["model"] if 'model' in checkpoint else checkpoint
        self.model.load_state_dict(model_state_load)

    def train(self):
        for epoch in range(0, 100):
            total_loss = 0.
            self.model.train()
            for batch, data in tenumerate(self.train_dataloader, desc='{} epoch'.format(epoch + 1)):
                id_labels = data["label"] if "label" in data else None
                self.optimizer.zero_grad()
                fis = self.model(data, id_labels)
                loss_ = fis["loss"].mean()
                total_loss += loss_.item()
                loss_.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.global_step += 1

            if (epoch) % 5 == 0 or (epoch + 1) == 100:
                _ = self.evaluate(epoch)
                if not os.path.isdir(self.model_save_dir):
                    os.makedirs(self.model_save_dir)
                model_state_dict = self.model.state_dict()
                checkpoint = {
                    "model": model_state_dict,
                }
                model_save_path = os.path.join(self.model_save_dir, f'{epoch}.pth')
                torch.save(checkpoint, model_save_path)

    def evaluate(self, epoch):
        self.model.eval()
        result_sum = []
        for eval_file_name, eval_dataloader in self.eval_dataloader_dic.items():
            self.eval_file_name = eval_file_name
            self.eval_task_name = self.eval_file_name.split("/")[-1].split('.')[0]
            vid_names = []
            all_features = []
            with torch.no_grad():
                for batch, data in tenumerate(eval_dataloader, desc=self.eval_task_name):
                    fis = self.model(data)
                    features = fis['id_feature']
                    all_features.extend(features.cpu().numpy())
                    vid_names.extend([str(vid_name) for vid_name in data["vid_name"]])

            result_sum.update(self.get_eer(all_features, vid_names, epoch))

        for key, value in result_sum.items():
            print('{}: {}'.format(key, value))
        self.model.train()
        return result_sum

    def get_eer(self, all_features, vid_names):
        name_feature_dic = dict(zip(vid_names, all_features))
        fin_csv = csv.reader(open(osp.join(self.eval_file_dir, self.eval_file_name)))
        features1 = []
        features2 = []
        pair_labels = []
        for i, row in enumerate(fin_csv):
            if i > 0:
                features1.append(name_feature_dic[row[0]])
                features2.append(name_feature_dic[row[1]])
                pair_labels.append(row[2] == '1')
        features1 = np.asarray(features1)
        features2 = np.asarray(features2)
        pair_labels = np.asarray(pair_labels)
        distances = 1 - np.sum(features1 * features2, axis=1) / (np.linalg.norm(features1, axis=1) * np.linalg.norm(features2, axis=1))
        min_dis = np.min(distances)
        max_dis = np.max(distances)
        accept_distances = distances[pair_labels == True]
        reject_distances = distances[pair_labels == False]
        FARs = []
        FRRs = []
        thresholds = []
        errors = []
        for threshold in np.linspace(min_dis, max_dis, num=1000):
            thresholds.append(threshold)
            FRR = np.sum(accept_distances >= threshold) / accept_distances.shape[0]
            FAR = np.sum(reject_distances < threshold) / reject_distances.shape[0]
            FRRs.append(FRR)
            FARs.append(FAR)
            errors.append(abs(FAR - FRR))
        min_errors_idx = np.argmin(np.asarray(errors))
        eer = (FRRs[min_errors_idx] + FARs[min_errors_idx]) / 2
        evaluate_result = {self.eval_task_name: eer}

        return evaluate_result
