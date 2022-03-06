import argparse
from src.utils import bool_flag

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="AdvNMT-backTrans")

    # for adv-nmt
    parser.add_argument('-langs_pair', type=str, default="zh-en", metavar='S',
                        help='dir of adv data')
    
    parser.add_argument('-part', type=str, default="", metavar='S',
                        help='part of adv data')
    
    parser.add_argument('-start', type=str, default="", metavar='S',
                        help='start of training')

    parser.add_argument('-end', type=str, default="", metavar='S',
                        help='end of training')

    # parser.add_argument('-job', type=str, default="200", metavar='S',
    #                     help='job of adv data, default with 4: 0, 1, 2, 3')

    parser.add_argument('-nmt', type=str, default="transformer", metavar='S',
                        help='type of training model')

    # parser.add_argument('-gpuid', type=int, default=0, metavar='N',
    #                     help='gpuid of translation model')

    parser.add_argument('-syn', type=str, default="synonym", metavar='S',
                        help='synonyms_obtain_method: wordnet or embedding')

    parser.add_argument('-alpha', type=float, default=1.0,
                        help='threshold of score')

    # main parameters
    parser.add_argument("--dump_path", type=str, default="./dumped/",
                        help="Experiment dump path")

    parser.add_argument("-exp_name", type=str, default="",
                        help="Experiment name")
    parser.add_argument("-job", type=str, default="0",
                        help="Experiment dataset")
    parser.add_argument("-exp_id", type=str, default="",
                        help="Experiment ID")

    parser.add_argument("-saliency", type=str, default="off",
                        help="use or not use word saliency, option: on , off")
    parser.add_argument("-saliencyReverse", type=str, default="off",
                        help="revese or not reverse word saliency, option: on , off")                    
    parser.add_argument("-greedy", type=str, default="off",
                        help="use of not use greedy replacement, option: on , off")

    parser.add_argument("-oracle", type=str, default="transformer")

    parser.add_argument("-ratio", type=float, default=0.2,
                        help="replacement ratio in each sentence")

    parser.add_argument("-togpu", type=str, default="on",
                        help="use of not use gpu, option: on , off")

    parser.add_argument("-dev_id", type=str, default="1",)

    parser.add_argument("-rand_ratio", type=float, default="0")
    
    # parser.add_argument("--save_periodic", type=int, default=0,
    #                     help="Save the model periodically (0 to disable)")

    params = parser.parse_args()
    params.saliency = bool_flag(params.saliency)
    params.saliencyReverse = bool_flag(params.saliencyReverse)
    params.greedy = bool_flag(params.greedy)
    params.togpu = bool_flag(params.togpu)

    if params.saliency:
        params.exp_id += '_saliency'
        if params.saliencyReverse:
            params.exp_id +='_reverse'
    else:
        params.exp_id += '_random_order'
    if params.rand_ratio == 0.0:
        params.exp_id += '_greedy'
    elif params.rand_ratio > 0 and params.rand_ratio < 1:
        params.exp_id += '_greedy{}'.format(str(int(10 * params.rand_ratio)))
    # elif params.rand_ratio == 1:
    #     params.exp_id += '_random'

    # if len(params.exp_id) == 0:
    #     params.exp_id += 'random'

    return params

if __name__ == "__main__":
    params = get_parser()