import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug

class Logger():
    def __init__(self, base_path,save_path):
        self.logger = open(os.path.join(base_path, save_path+".txt"), 'w')

    def __call__(self, string, end='\n', print_=True):
        if print_:
            print("{}".format(string), end=end)
            if end == '\n':
                self.logger.write('{}\n'.format(string))
            else:
                self.logger.write('{} '.format(string))
            self.logger.flush()
            
def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=50, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='SS', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--Iteration', type=int, default=20000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=1.0, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='/root/autodl-tmp/DC_DSA_DM/result/', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--log_path', type=str, default='ours', help='distance metric')
    parser.add_argument('--eta', type=int, default=0.5, help='threshold for gradient accent')
    
    
    args = parser.parse_args()
    args.method = 'DM'
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True
    
    logger = Logger(args.save_path,args.log_path)
    logger(f"Save dir: {os.path.join(args.save_path, args.log_path)}")
    
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    eval_it_pool = np.arange(0, args.Iteration+1, 2000).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    logger('eval_it_pool: {}'.format(eval_it_pool))
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    # print(channel,im_size)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)


    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    for exp in range(args.num_exp):
        logger('\n================== Exp %d ==================\n '%exp)
        logger('Hyper-parameters: \n {}'.format(args.__dict__))
        logger('Evaluation model pool: {}'.format(model_eval_pool))

        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        attr_all = []
        indices_class = [[] for c in range(num_classes)]

        # images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))] ###
        # labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        images_all = [torch.unsqueeze(dst_train[0][i], dim=0) for i in range(len(dst_train[0]))]
        labels_all = [dst_train[1][i] for i in range(len(dst_train[1]))]
        attr_all = [dst_train[2][i] for i in range(len(dst_train[2]))]

        images_test_all = [torch.unsqueeze(dst_test[0][i], dim=0) for i in range(len(dst_test[0]))]
        labels_test_all = [dst_test[1][i] for i in range(len(dst_test[1]))]

        for i, lab in enumerate(labels_all):
            # indices_class[lab].append(i) ###
            indices_class[int(lab)].append(i)
        images_all = torch.cat(images_all, dim=0)
        labels_all = torch.tensor(labels_all, dtype=torch.long)
        attr_all = torch.tensor(attr_all, dtype=torch.float, device=args.device)
        images_test_all = torch.cat(images_test_all, dim=0).to(args.device)
        labels_test_all = torch.tensor(labels_test_all, dtype=torch.long, device=args.device)
        train_dataset = TensorDataset(images_all, labels_all)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_real,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        train_test_dataset = copy.deepcopy(train_dataset)
        train_test_loader = DataLoader(
            train_test_dataset,
            batch_size=args.batch_real,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        attr_dims = []
        attr_dims.append(torch.max(labels_all).item() + 1)
        attr_dims.append(torch.max(attr_all).item() + 1)
        for c in range(num_classes):
            logger('class c = %d: %d real images'%(c, len(indices_class[c])))

        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        for ch in range(channel):
            logger('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

        
        ''' training '''
        model_b1 = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
        model_b2 = get_network(args.model, channel, num_classes, im_size).to(args.device)
        logger('%s training begins'%get_time())
        optimizer_b1 = torch.optim.Adam(
            model_b1.parameters(),
            lr=0.001,
            weight_decay=0,
        )
        optimizer_b2 = torch.optim.Adam(
            model_b2.parameters(),
            lr=0.001,
            weight_decay=0,
        )
        count_iter = 0
        scores = []
        probs = []
        for it in range(50):
            logger('{}: Epoch {}'.format(get_time(),it))
            model_b1.train()
            model_b2.train()
            for _ in range(len(train_dataset)//args.batch_real):
                # train_iter = iter(train_loader)
                # images, labels = next(train_iter)
                try:
                   images, labels = next(train_iter)
                except:
                    train_iter = iter(train_loader)
                    images, labels = next(train_iter)
                images, labels = images.to(args.device), labels.to(args.device) # [256,3,28,28], [256,2]
                logit1 = model_b1(images) # [256, 10]
                loss_per_sample1 = F.cross_entropy(logit1, labels, reduction='none')
                p1 = torch.exp(-loss_per_sample1.detach())
                for c in range(attr_dims[0]):
                    p1[labels==c] /= p1[labels==c].max()

                logit2 = model_b2(images)
                loss_per_sample2 = F.cross_entropy(logit2, labels, reduction='none')
                p2 = torch.exp(-loss_per_sample2.detach())
                for c in range(attr_dims[0]):
                    p2[labels==c] /= p2[labels==c].max()

                loss_weight1 = torch.zeros_like(loss_per_sample1)
                loss_weight1[(p1>args.eta) & (p2>args.eta)] = 1.0
                if it>0:
                    loss_weight1[(p1>args.eta) & (p2<args.eta)] = -0.0001

                loss_weight2 = torch.zeros_like(loss_per_sample2)
                loss_weight2[(p1>args.eta) & (p2>args.eta)] = 1.0
                if it>0:
                    loss_weight2[(p1<args.eta) & (p2>args.eta)] = -0.0001

                loss1 = (loss_weight1 * loss_per_sample1).mean()
                optimizer_b1.zero_grad()
                loss1.backward()
                optimizer_b1.step()

                loss2 = (loss_weight2 * loss_per_sample2).mean()
                optimizer_b2.zero_grad()
                loss2.backward()
                optimizer_b2.step()

                count_iter += 1
            model_b1.eval()
            model_b2.eval()
            score = []
            prob = []
            for images, labels in train_test_loader:
                images, labels = images.to(args.device), labels.to(args.device) # [256,3,28,28], [256,2]
                with torch.no_grad():
                    logit1 = model_b1(images)
                    logit2 = model_b2(images)
                    loss1 = F.cross_entropy(logit1, labels, reduction='none')
                    loss2 = F.cross_entropy(logit2, labels, reduction='none')
                    p1 = F.softmax(logit1, dim=1) # [256,10]
                    p2 = F.softmax(logit2, dim=1)
                    p = 0.5*p1 + 0.5*p2
                    pred = p.data.max(1, keepdim=True)[1].squeeze(1)
                    score.append(
                        0.5*torch.exp(-loss1) + 0.5*torch.exp(-loss2)
                    )
                    prob.append(
                        p
                    )
            score = torch.cat(score)
            scores.append(score)
            prob = torch.cat(prob)
            probs.append(prob)

        model_path = os.path.join(args.save_path, "CIFAR10_model_b_0.9.th")
        state_dict = {
            'scores': torch.stack(scores).t().cpu(),
            'probs': probs,
        }
        torch.save(state_dict, model_path)

    logger('\n==================== Final Results ====================\n')



if __name__ == '__main__':
    main()


