"""
    Main program
"""


from utils.metrics import purity, info_score
import time
from argparse import ArgumentParser
from utils.data_preprocessor import get_dataset
from utils.trainers import TrainerClusterwise
from models.RNN import RNNMultiplePointProcesses
from utils.file_system_utils import create_folder
import torch
import pickle
import json

import numpy
import numpy as np
def parse_arguments():
    """
        Processes and returns cmd arguments
        inputs:
                None
        outputs:
                args
    """
    parser = ArgumentParser()
    parser.add_argument('--path_to_files', type=str, required=True, help='path to data, required')
    parser.add_argument('--n_steps', type=int, default=128, help='number of steps in partitions, default - 128')
    parser.add_argument('--n_clusters', type=int, required=True, help='initial number of clusters, required')
    parser.add_argument('--n_classes', type=int, required=True, help='number of types of events, required')
    parser.add_argument('--hidden_size', type=int, default=128, help='RNN hidden size, default - 128')
    parser.add_argument('--num_layers', type=int, default=3, help='number of RNN layers, default - 3')
    parser.add_argument('--dropout', type=float, default=0.2, help='RNN dropout rate, default - 0.3')
    parser.add_argument('--lr', type=float, default=0.1, help='optimizer initial learning rate, default - 0.1')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='optimizer weight decay, default - 1e-5')
    parser.add_argument('--n_runs', type=int, default=5, help='number of starts, default - 5')
    parser.add_argument('--save_dir', type=str, required=True, help='saves results to experiments/save_dir, required')
    parser.add_argument('--save_best_model', type=bool, default=True, help='if True, saves the state of the best '
                                                                           'model according to loss')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='is used for log-s regularization log(x) -> log(x '
                                                                    '+ epsilon), default - 1e-8')
    parser.add_argument('--max_epoch', type=int, default=20, help='number of epochs of EM algorithm, default - 50')
    parser.add_argument('--max_m_step_epoch', type=float, default=10, help='int(max_m_step_epoch) - number of epochs'
                                                                           ' of neural net training on '
                                                                           'M-step, default - 10')
    parser.add_argument('--lr_update_tol', type=int, default=25, help='tolerance before updating learning rate, '
                                                                      'default - 25')
    parser.add_argument('--lr_update_param', type=float, default=0.5, help='learning rate multiplier, default - 0.5')
    parser.add_argument('--min_lr', type=float, help='if provided, defines the minimal available value of lr, '
                                                     'if achieved updates to updated_lr')
    parser.add_argument('--updated_lr', type=float, default=0.001, help='updates lr to it when min_lr is achieved')
    parser.add_argument('--batch_size', type=int, default=150, help='batch size during neural net training, default - '
                                                                   '50')
    parser.add_argument('--verbose', type=bool, default=True, help='if true, prints logs, default - True')
    parser.add_argument('--device', type=str, default='cuda:1', help='device that should be used for training, default - '
                                                                  'cpu')
    parser.add_argument('--max_computing_size', type=int, help='if provided, constraints the max number of processing '
                                                               'point in one step of EM algorithm')
    parser.add_argument('--full_purity', type=bool, default=False, help='if true, uses all dataset to compute purity')
    parser.add_argument('--random_walking_max_epoch', type=int, default=10, help='the epoch before enforcing number of'
                                                                                 ' clusters, default - 40')
    parser.add_argument('--true_clusters', type=int, required=True, help='true number of clusters, required')
    parser.add_argument('--upper_bound_clusters', type=int, default=10, help='upper bound of the number of clusters'
                                                                             ' during random walking, default - 10')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_arguments()

    # reading datasets
    if args.verbose:
        print('Reading dataset')
    data, target = get_dataset(args.path_to_files, args.n_classes, args.n_steps)
    if args.verbose:
        print('Dataset is loaded')

    # preparing folders
    if args.verbose:
        print('Preparing folders')
    create_folder('experiments')
    create_folder('experiments/RNN/' + args.save_dir)
    path_to_results = 'experiments/RNN/' + args.save_dir
    times = np.zeros(args.n_runs)
    pred = torch.zeros(args.n_runs, list(data.size())[0])
    labels = np.zeros((args.n_runs, list(data.size())[0]))
    # iterations over runs
    i = 0
    while i < args.n_runs:
        time_start = time.clock()
        if args.verbose:
            print('Run {}/{}'.format(i + 1, args.n_runs))
        model = RNNMultiplePointProcesses(args.n_classes + 1, args.hidden_size, args.num_layers, args.n_classes,
                                           args.n_clusters, args.n_steps, dropout=args.dropout).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_model_path = path_to_results + '/exp_{}'.format(i) + '/best_model.pt'
        create_folder(path_to_results + '/exp_{}'.format(i))
        exp_folder = path_to_results + '/exp_{}'.format(i)
        trainer = TrainerClusterwise(model, optimizer, args.device, data, args.n_clusters, target=target,
                                     epsilon=args.epsilon, max_epoch=args.max_epoch,
                                     max_m_step_epoch=args.max_m_step_epoch, lr=args.lr,
                                     random_walking_max_epoch=args.random_walking_max_epoch,
                                     true_clusters=args.true_clusters, upper_bound_clusters=args.upper_bound_clusters,
                                     lr_update_tol=args.lr_update_tol, lr_update_param=args.lr_update_param,
                                     min_lr=args.min_lr, updated_lr=args.updated_lr,
                                     batch_size=args.batch_size, verbose=args.verbose,
                                     best_model_path=best_model_path if args.save_best_model else None,
                                     max_computing_size=args.max_computing_size, full_purity=args.full_purity)
        losses, results, cluster_part, stats, clusters_pred  = trainer.train()
        pred[i,:] = clusters_pred
        # results check
        if cluster_part is None:
            if args.verbose:
                print('Solution failed')
            continue

        # saving results
        with open(exp_folder + '/losses.pkl', 'wb') as f:
            pickle.dump(losses, f)
        with open(exp_folder + '/results.pkl', 'wb') as f:
            pickle.dump(results, f)
        with open(exp_folder + '/stats.pkl', 'wb') as f:
            pickle.dump(stats, f)
        with open(exp_folder + '/args.json', 'w') as f:
            json.dump(vars(args), f)
                    
        labels1 = {"labels": clusters_pred.cpu().detach().numpy()}
        with open(exp_folder + '/labels.csv', 'w') as f:
            f.write("%s\n"%(labels1))
        torch.save(trainer.model, exp_folder + '/last_model.pt')
        
        times[i] = time.clock() - time_start
        i += 1
    clusters = numpy.zeros(list(data.size())[0])
    for k in range (list(data.size())[0]):
        clus = pred[:,k].cpu().detach().numpy()
        labels[:, k] = clus
        clusters[k] = max(set(list(clus)), key = list(clus).count)    
    #info = info_score(torch.from_numpy(clusters),
                      #target.to('cpu'), len(numpy.unique(target.to('cpu'))))
    pur_val_mean = np.mean([purity(torch.from_numpy(x), target.to('cpu')) for x in labels])
    pur_val_std = np.std([purity(torch.from_numpy(x), target.to('cpu')) for x in labels])
    time_mean = np.mean(times*10)
    time_std = np.std(times*10)
    res = { "Purity": f'{pur_val_mean:.4f}+-{pur_val_std:.4f}' ,
            "Mean run time": f'{time_mean:.4f}+-{time_std:.4f}', "labels": clusters}
    with open(f'{path_to_results}/results.csv', 'w') as f:
        for key in res.keys():
            f.write("%s,%s\n"%(key,res[key]))