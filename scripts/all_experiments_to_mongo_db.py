import argparse
import os
import sys

import experiment_to_mongo_db

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'src')))
from utils.metadata import LOG_DIRECTORY


def main(args):

    for root, folders, files in os.walk(LOG_DIRECTORY):
        if os.path.isdir(os.path.join(root, '_sources')):
            args.exp_name = os.path.relpath(root, LOG_DIRECTORY)
            experiment_to_mongo_db.main(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dbhost', type=str, default='localhost')
    parser.add_argument('--dbport', type=int, default=27017)
    parser.add_argument('--dbuser', type=str, default=None)
    parser.add_argument('--dbpassword', type=str, default=None)
    args = parser.parse_args()
    main(args)
