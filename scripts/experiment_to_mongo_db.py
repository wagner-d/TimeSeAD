"""
Inserts the results of an experiment into a running mongo DB instance.
"""

import argparse
import json
import os
import sys
from urllib.parse import quote_plus

import dateutil.parser
import gridfs
import pymongo
from sacred.dependencies import get_digest
from sacred.observers import MongoObserver

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'src')))
from utils.metadata import LOG_DIRECTORY


def save_resource(fs, filename, file_location):
    md5hash = get_digest(file_location)
    if fs.exists(filename=filename, md5=md5hash):
        return

    with open(file_location, "rb") as f:
        fs.put(f, filename=filename)

    return md5hash


def save_artifact(fs, base_dir, filename, run_id, exp_name, metadata=None, content_type=None):
    with open(os.path.join(base_dir, filename), 'rb') as f:
        db_filename = f'artifact://{exp_name}/{run_id}/{filename}'
        if content_type is None:
            content_type = MongoObserver._try_to_detect_content_type(filename)

        file_id = fs.put(
            f, filename=db_filename, metadata=metadata, content_type=content_type
        )

    return file_id


def save_source(fs, exp_dir, filename, file_location):
    with open(os.path.join(exp_dir, file_location), 'rb') as f:
        file_id = fs.put(f, filename=filename)

    return file_id


def sanitize_database_name(name: str) -> str:
    if len(name) == 0 or len(name) > 64:
        raise ValueError('MongoDB database name must have between 1 and 64 characters (inclusive)!')

    # Path separators
    name = name.replace('/', '::')
    name = name.replace('\\', '::')

    # Other disallowed characters
    name = name.replace('"', '_')
    name = name.replace('$', '_')
    name = name.replace('.', '_')
    name = name.replace(' ', '_')
    name = name.replace('\0', '_')

    # Omniboard will have difficulties with characters that have a special meaning in URLs
    name = name.replace('#', '_')
    name = name.replace('?', '_')
    name = name.replace('&', '_')

    # Additionally, MongoDB on Windows does not like the following characters
    # name = name.replace('*', '_')
    # name = name.replace('<', '_')
    # name = name.replace('>', '_')
    # name = name.replace(':', '_')
    # name = name.replace('|', '_')

    return name


def main(args):
    exp_dir = os.path.abspath(os.path.join(LOG_DIRECTORY, args.exp_name))

    if args.dbuser is None:
        mongo_uri = f'mongodb://{args.dbhost}:{args.dbport}/'
    else:
        mongo_uri = f'mongodb://{quote_plus(args.dbuser)}:{quote_plus(args.dbpassword)}@{args.dbhost}:{args.dbport}/'

    client = pymongo.MongoClient(mongo_uri)

    for subdir, dirs, files in os.walk(exp_dir):
        if 'run.json' not in files:
            continue

        with open(os.path.join(subdir, 'run.json'), encoding='utf8') as f:
            run = json.load(f)

        run_id = int(os.path.basename(subdir))

        args.exp_name = sanitize_database_name(args.exp_name)
        database = client[args.exp_name]

        entry = database['runs'].find_one({'_id': run_id})
        if entry is not None:
            # An entry with the same id already exists in the database
            # If the start date matches, we will skip this entry, otherwise we throw an error
            # Note that MongoDB truncates timestamps to milliseconds, so we have to do the same before comparing
            start_time = dateutil.parser.parse(run['start_time'])
            start_time = start_time.replace(microsecond=(start_time.microsecond // 1000) * 1000)
            if entry['start_time'] != start_time:
                raise ValueError(f'A different entry with the same id ({run_id:d}) already exists in the DB!')

            print(f'Skipping entry with id {run_id:d}. Already exists in the database.')
            continue

        with open(os.path.join(subdir, 'config.json'), encoding='utf8') as f:
            config = json.load(f)

        with open(os.path.join(subdir, 'metrics.json'), encoding='utf8') as f:
            metrics = json.load(f)

        with open(os.path.join(subdir, 'cout.txt'), encoding='utf8') as f:
            cout = f.read()

        if os.path.isfile(os.path.join(subdir, 'info.json')):
            with open(os.path.join(subdir, 'info.json'), encoding='utf8') as f:
                info = json.load(f)
        else:
            info = {}


        metrics_collection = [
            {'name': metric,
             'run_id': run_id,
             'steps': m_value['steps'],
             'timestamps': [dateutil.parser.parse(date) for date in m_value['timestamps']],
             'values': m_value['values']
             } for metric, m_value in metrics.items()
        ]

        run.update(
            _id=run_id,
            config=config,
            captured_out=cout,
            metrics=metrics
        )

        collection = database['metrics']
        fs = gridfs.GridFS(database)

        metrics_list = []

        for metric_dict in metrics_collection:
            ret = collection.insert_one(metric_dict)
            metrics_list.append({'name': metric_dict['name'], 'id': ret.inserted_id})

        info.update(metrics=metrics_list)

        run['artifacts'] = [{'name': filename, 'file_id': save_artifact(fs, subdir, filename, run_id, args.exp_name)}
                            for filename in run['artifacts']]
        run['resources'] = [(filename, save_resource(fs, filename, file_location))
                            for filename, file_location in run['resources']]
        run['experiment']['sources'] = [(filename, save_source(fs, exp_dir, filename, file_location))
                                        for filename, file_location in run['experiment']['sources']]

        runs_collection = {
            '_id': run_id,
            'experiment': run['experiment'],
            'format': 'MongoObserver-0.7.0',
            'command': run['command'],
            'host': run['host'],
            'start_time': dateutil.parser.parse(run['start_time']),
            'stop_time': dateutil.parser.parse(run['stop_time']) if 'stop_time' in run else None,
            'heartbeat': dateutil.parser.parse(run['heartbeat']),
            'config': config,
            'meta': run['meta'],
            'status': run['status'],
            'artifacts': run['artifacts'],
            'resources': run['resources'],
            'result': run['result'],
            'captured_out': cout,
            'info': info
        }

        collection = database['runs']
        collection.insert_one(runs_collection)

    client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', type=str)
    parser.add_argument('--dbhost', type=str, default='localhost')
    parser.add_argument('--dbport', type=int, default=27017)
    parser.add_argument('--dbuser', type=str, default=None)
    parser.add_argument('--dbpassword', type=str, default=None)
    args = parser.parse_args()
    main(args)
