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
    
    params.exp_name = 'en_de_en/transformer/rogr/' + 'job'+str(params.job)
    params.saliency=False

    print('params.exp_name', params.exp_name)

    params.oracle = 'transformer'
    params.nmt = "transformer"


    logger = initialize_exp(params)
    # logger saved in params.dump_path
    logger.info("Src path: " + src_path)
    logger.info("Tgt path: " + tgt_path)
    # logger.info("Current gpu: " + str(params.gpuid))
    logger.info('Current syn: ' + params.syn)
    logger.info('Current alpha: '+str(params.alpha))
    logger.info('Current model: '+params.nmt)
    logger.info('Current oracle: '+params.oracle)
    logger.info('Current rand_ration: {}'.format(params.rand_ratio))

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
        togpu=params.togpu
    )

    AdvNMT.attack_forward()
    #CUDA_VISIBLE_DEVICES