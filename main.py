# system libs
import os
import time
from os.path import join, splitext

# libs
import argparse
import yaml

# custom libs
import src.myTools as mts
from src.makeup import TeethSeg

# global variables
today = time.strftime("%y-%m-%d", time.localtime(time.time()))
CD = os.path.dirname( os.path.abspath(__file__) )

def get_args():
    parser = argparse.ArgumentParser(description='Individual tooth segmentation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-p", "--pseudo_er", dest="pseudo_er",
                             required=False, action='store_true',
                             help="Network inference making pseudo edge region")
    parser.add_argument("-c", "--init_contours", dest="inits",
                             required=False, action='store_true',
                             help="Obtain initial contours")
    parser.add_argument("-s", "--snake", dest="snake",
                             required=False, action='store_true',
                             help="Snake; active contour evolution")
    parser.add_argument("-i", "--id_region", dest="id_region",
                             required=False, action='store_true',
                             help="Identification of regions")
    parser.add_argument("-A", "--ALL", dest="ALL",
                             required=False, action='store_true',
                             help="Do every process in a row")
    parser.add_argument("--cfg", dest="path_cfg", type=str, default='config/default.yaml',
                             required=False, metavar="CFG",
                             help="configuration file")
    parser.add_argument("--data-dir", dest="data_dir", type=str, default='',
                              required=False,
                              help="the data dir folder (same as DATA:DIR: in config)")
    parser.add_argument("--out-dir", dest="out", type=str, default='',
                            required=False,
                            help="the output folder")
    parser.add_argument("--image", dest="image", type=str, default='',
                            required=False,
                            help="specify one image only")
    parser.add_argument("--root", dest="root", required=False, action='store_true',
                            help="set the DEFALUT:ROOT as the path of main.py (%s)" % CD)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.ALL:
        args.pseudo_er = True
        args.inits = True
        args.snake = True
        args.id_region = True

    # force config to be resolved with CD if args.root is true and config path
    # is not absolute
    path_cfg = args.path_cfg
    if path_cfg[0] != '/' and args.root:
        path_cfg = os.path.join(CD, path_cfg)

    with open(path_cfg, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.root:
        config['DEFAULT']['ROOT'] = CD

    ROOT = config['DEFAULT']['ROOT']
    dir_image = join(ROOT, config['DATA']['DIR']) if not args.data_dir else args.data_dir
    dir_output = join(ROOT, *[config['EVAL']['DIR'], f'{today}/'])
    mts.makeDir(join(ROOT, config['EVAL']['DIR']))
    mts.makeDir(dir_output)

    if args.image:
        image = os.path.abspath(args.image)
        image = args.image
        dir_image = os.path.dirname(image)
        number = int(''.join([ x for x in os.path.basename(image) if x.isdigit()]))
        imgs = [number]
        config['DATA']['DIR'] = dir_image
        # config['EVAL']['DIR'] = dir_image
        # dir_output = dir_image+'/'
    else:
        imgs = [int(splitext(file)[0]) for file in os.listdir(dir_image) if splitext(file)[-1][1:] in config['DATA']['EXT']]

    # imgs = [int(splitext(file)[0]) for file in os.listdir(dir_image) if splitext(file)[-1][1:] in config['DATA']['EXT']]
    from pprint import pprint
    pprint(config)
    print(args.root, CD)


    for ni in imgs:
        dir_img = join(dir_output, '%05d/' % ni)
        # dir_img = dir_output
        mts.makeDir(dir_img)
        sts = mts.SaveTools(dir_img)
        print(dir_output)

        # Inference pseudo edge-regions with a deep neural network
        ts = TeethSeg(dir_img, ni, sts, config)
        if args.pseudo_er: ts.pseudoER()
        if args.inits: ts.initContour()
        if args.snake: ts.snake()
        if args.id_region: ts.tem()

        if args.image:
            os.system('mv "%s/output.png" "%s/%06d.outline.png"' % (dir_img, config['DATA']['DIR'], ni) )
            os.system('rm -rf "%s"' % dir_img )
