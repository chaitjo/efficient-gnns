import os
import torch
import numpy as np
import argparse


if __name__ == "__main__":
    # Experiment settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--expt_name', type=str, default="debug",
                        help='Name of experiment logs/results folder')
    args = parser.parse_args()
    
    expt_args = []
    model = []
    total_param = []
    BestEpoch = []
    Validation = []
    Test = []
    Train = []
    BestTrain = []

    for root, dirs, files in os.walk(f'logs/{args.expt_name}'):
        if 'results.pt' in files:
            results = torch.load(os.path.join(root, 'results.pt'), map_location=torch.device('cpu'))
            expt_args.append(results['args'])
            total_param.append(results['total_param'])
            BestEpoch.append(results['BestEpoch'])
            Train.append(results['Train'])
            Validation.append(results['Validation'])
            Test.append(results['Test'])

            print(results['args'].seed, results['Test'], results['Validation'])
            
    print(expt_args[0])
    print()
    print(f'Test performance: {np.mean(Test)*100:.2f} +- {np.std(Test)*100:.2f}')
    print(f'Validation performance: {np.mean(Validation)*100:.2f} +- {np.std(Validation)*100:.2f}')
    print(f'Train performance: {np.mean(Train)*100:.2f} +- {np.std(Train)*100:.2f}')
    print(f'Total parameters: {int(np.mean(total_param))}')