# -*- coding: utf-8 -*-
import os
import sys
import argparse
import os.path as osp
import torch
import numpy as np
from termcolor import cprint

DIR = osp.realpath(osp.dirname(__file__))
sys.path.insert(0, DIR)
sys.path.insert(0, osp.join(DIR, 'src/'))
sys.path.insert(0, osp.join(DIR, '/'))

from src.main.trainer import Trainer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_file", type=str, default='none', help="address of configurtion file")
    parser.add_argument("--mode", type=str, default='train', help="train or evaluate")
    args = parser.parse_args()
    return args


def parse_conf(conf_path):
    conf_dict = {}
    with open(conf_path, 'r', encoding='utf-8') as filein:
        for line in filein:
            line = line.strip()
            if len(line) == 0 or line[0] == "#" or line[0] == "[":
                continue
            line = line.split("#")[0].strip()
            data = line.split("=")
            assert len(data) == 2, data
            key = data[0].strip()
            values = data[1].strip()
            conf_dict[key] = values

    # bool
    for key, value in conf_dict.items():
        if ',' in value:
            value = [v.strip() for v in value.split(',')]
            try:
                value.remove('')
            except ValueError:
                pass
        else:
            value = value.strip()
        if value == "True" or value == "T":
            value = True
        if value == "False" or value == "F":
            value = False
        conf_dict[key] = value
    try:
        conf_dict['modality'] = [conf_dict['modality']] if isinstance(conf_dict['modality'], str) else conf_dict['modality']
        conf_dict["frames_root"] = [conf_dict["frames_root"]] if isinstance(conf_dict["frames_root"], str) else conf_dict["frames_root"]
    except KeyError:
        print(KeyError)
    return conf_dict


if __name__ == "__main__":
    args = get_args()
    conf_file = args.conf_file
    conf = parse_conf(conf_file)
    conf["mode"] = args.mode
    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda", 0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")
    conf["device"] = device

    task_type = conf["task_name"].split("_")[0]

    if args.mode == "train":
        trainer = Trainer(conf)
        trainer.train()

    elif args.mode == "evaluate":
        trainer = Trainer(conf)
        parm_dir = conf["model_save_dir"]
        pretrained_parms_name_list = os.listdir(parm_dir)
        pretrained_parms_name_list.sort()
        for i in range(len(pretrained_parms_name_list)):
            parm_path = os.path.join(parm_dir, pretrained_parms_name_list[i].strip())
            trainer.reload(parm_path)
            result_sum = trainer.evaluate(trainer.start_epoch)

    else:
        raise NotImplementedError
