import argparse
import json
from urllib.parse import quote_plus

import pymongo


DEFAULT_MONGO_DATABASES = {'admin', 'config', 'local'}


def main(args):
    if args.dbuser is None:
        mongo_uri = f'mongodb://{args.dbhost}:{args.dbport}'
    else:
        mongo_uri = f'mongodb://{quote_plus(args.dbuser)}:{quote_plus(args.dbpassword)}@{args.dbhost}:{args.dbport}'

    client = pymongo.MongoClient(mongo_uri)

    databases = set(client.list_database_names())
    client.close()

    # Remove standard mongo databases
    databases = databases.difference(DEFAULT_MONGO_DATABASES)

    db_config = {
        db_name: {
            'mongodbURI': f'{mongo_uri}/{db_name}?authSource=admin',
            'path': f'/{db_name}'
        }
        for db_name in databases
    }

    with open(args.out_file, mode='w') as f:
        json.dump(db_config, f, indent=2)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Parameters to connect to the database and retrieve a list of databases
    parser.add_argument('--out_file', type=str, default='db_config.json')
    parser.add_argument('--dbhost', type=str, default='localhost')
    parser.add_argument('--dbport', type=int, default=27017)
    parser.add_argument('--dbuser', type=str, default=None)
    parser.add_argument('--dbpassword', type=str, default=None)

    args = parser.parse_args()
    main(args)

