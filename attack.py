"""Attack loop
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
import tqdm

import torch
import torchvision
import torch.utils.data as data

from scipy.misc import imsave
from dataset import Dataset, default_inception_transform


def run_attack(args, attack):
    assert args.input_dir
    print('Start Attack!')
    device = 'cuda' if not args.no_gpu else 'cpu'

    dataset = Dataset(args, transform=default_inception_transform(args.img_size))

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True)

    model = torchvision.models.inception_v3(pretrained=False, transform_input=False).to(device)

    if args.checkpoint_path is not None and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("Error: No checkpoint found at %s." % args.checkpoint_path)

    model.eval()

    for batch_idx, (input, target, true_class, img_name) in enumerate(loader):
        input = input.to(device)
        target = target.to(device)
        true_class = true_class.to(device)

        input_adv = attack.run(model, input, target, true_class, img_name, batch_idx)

        start_index = args.batch_size * batch_idx
        indices = list(range(start_index, start_index + input.size(0)))
        for filename, o in zip(dataset.filenames(indices), input_adv):
            output_file = os.path.join(args.output_dir, os.path.basename(filename))
            imsave(output_file, (o + 1.0) * 0.5, format='png')


def eval_attack(args):
    print('Evaluation')
    device = 'cuda' if not args.no_gpu else 'cpu'
    dataset = Dataset(args, args.output_dir, transform=default_inception_transform(args.img_size))
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False)

    model = torchvision.models.inception_v3(pretrained=False, transform_input=False).to(device)

    if args.checkpoint_path is not None and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("Error: No checkpoint found at %s." % args.checkpoint_path)

    model.eval()
    correct_true = 0
    correct_target = 0
    column = ['ImageId', 'Trueclass', 'Fooledclass', 'Targetedclass']
    df = pd.DataFrame(columns=column, index=range(len(loader)))
    tqdm_test = tqdm.tqdm(loader, ncols=100)
    for batch_idx, (input, target, true_class, img_name) in enumerate(tqdm_test):
        input = input.to(device)
        output = model(input)
        prediction = torch.argmax(output, dim=1)
        correct_true += 1 if prediction.item() == true_class.item() else 0
        correct_target += 1 if prediction.item() == target.item() else 0

        df.iloc[batch_idx] = [img_name, true_class.item(), prediction.item(), target.item()]

    df.to_csv(os.path.join(args.output_dir, 'result.csv'), sep=',')
    print('GT Accuracy : {:.3f}'.format(correct_true / len(loader) * 100))
    print('Target Accuracy : {:.3f}'.format(correct_true / len(loader) * 100))