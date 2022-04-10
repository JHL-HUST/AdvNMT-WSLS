from src.utils import initialize_exp
from my_parser import get_parser
from attacker.AttackerContainer import AttackerWrapper
import os

if __name__ == "__main__":
    params = get_parser()

    src_path = './corpus/wmt19/jobs/job' + str(params.job) + '.test.en'

    print('src_path', src_path)

    tgt_path = './corpus/wmt19/jobs/job' + str(params.job) + '.test.de'
    print('tgt_path', tgt_path)


    params.exp_name = 'en_de_en/transformer/wsls/'+ 'job'+str(params.job)


    print('params.exp_name', params.exp_name)
    params.saliency = True

    params.oracle = 'transformer'
    params.nmt = "transformer"


    logger = initialize_exp(params)
    # logger saved in params.dump_path
    # logger.info("Using the {} for translating...".format(params.nmt))
    logger.info("Src path: " + src_path)
    logger.info("Tgt path: " + tgt_path)

    logger.info('Current syn: ' + params.syn)
    logger.info('Current alpha: '+str(params.alpha))
    logger.info('Current model: '+params.nmt)
    logger.info('Current oracle: '+params.oracle)
    logger.info('Current rand_ration: {}'.format(params.rand_ratio))

    print('wsls init path: ', "./dumped/en_de_en/transformer/gogr_res/job{}/data/".format(params.job))

    AdvNMT = AttackerWrapper(
        src_path=src_path,
        ref_path=tgt_path,
        translate_model_type=params.nmt,
        synonyms_obtain=params.syn,
        langs_pair=params.langs_pair,
        log_path=params.dump_path, alpha=params.alpha, logger=logger,
        saliency=params.saliency,
        saliencyReverse=params.saliencyReverse,
        oracle=params.oracle,
        random_ratio=params.rand_ratio,
        togpu=params.togpu,
        init_dir="./dumped/en_de_en/transformer/gogr_res/job{}/data/".format(params.job)
    )

    AdvNMT.attack_wsls()