import argparse
from utils import Params
import os
from director import Director



def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str) 
    parser.add_argument('--model_dir', '--md', 
                        default='C:\\Users\\aisat\\Projet SIA\\seq2seq-git\\experiments\\attention\\e2f',
                        type=str)
    
    parser.add_argument('--mode', '--m', default='train', type=str)
    parser.add_argument('--resume', '--r', action='store_true')
    parser.add_argument('--heatmap', '--hm', action='store_true') 
    args = parser.parse_args() 
    
    json_path = os.path.join(args.model_dir, 'config.json')
    print(json_path)
    if not os.path.exists(json_path):
        raise FileNotFoundError('there is no config json file') 
    params_hps = Params(json_path)
    params_hps.dict.update(args.__dict__)
    if params_hps.mode == 'test':
        params_hps.batch_size = 1
    print(params_hps)
    return params_hps


def main():
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    args = get_parameters() #params_hps
    director = Director(args)
    if args.mode == 'train':
        director.train()
    elif args.mode == 'test':
        director.test()
    else:
        raise ValueError('mode is illegal, has to be one of train or test')


if __name__ == '__main__':
    main()
