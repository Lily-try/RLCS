import logging

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    if filename:
        fh = logging.FileHandler(filename, "w",encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def get_log_path(logroot,args):
    if args.attack == 'meta':
        return f'{logroot}{args.dataset}_{args.attack}_{args.ptb_rate}_{args.method}.log'
    elif args.attack == 'random':
        return f'{logroot}{args.dataset}_{args.attack}_{args.type}_{args.ptb_rate}_{args.method}.log'
    elif args.attack == 'add':
        return f'{logroot}{args.dataset}_{args.attack}_{args.noise_level}_{args.method}.log'
    elif args.attack == 'del':
        return f'{logroot}{args.dataset}_{args.attack}_{args.ptb_rate}_{args.method}.log'
    else:
        return f'{logroot}{args.dataset}_{args.method}.log'
