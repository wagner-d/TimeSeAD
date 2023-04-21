import collections.abc
import functools
import inspect
import os
import tempfile
import types
from typing import Any, Sequence, Dict, Callable, Optional, Tuple

import gridfs
import pymongo
import sacred.utils
from jsonpickle.unpickler import _IDProxy
from sacred import Ingredient, Experiment, SETTINGS
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.run import Run
from sacred.utils import apply_backspaces_and_linefeeds

from timesead.utils.metadata import PROJECT_ROOT, LOG_DIRECTORY, DISABLE_NVIDIA_SMI
from timesead.utils.sys_utils import check_path


def make_experiment(exp_name: str = None, ingredients: Sequence[Ingredient] = ()) -> Experiment:
    SETTINGS.DISCOVER_SOURCES = "sys"
    SETTINGS.HOST_INFO.INCLUDE_GPU_INFO = not DISABLE_NVIDIA_SMI

    if exp_name is None:
        exp_name, _ = os.path.splitext(os.path.basename(inspect.stack()[1].filename))

    experiment = Experiment(exp_name, base_dir=PROJECT_ROOT, ingredients=ingredients)

    experiment_path = os.path.abspath(os.path.join(LOG_DIRECTORY, exp_name))
    check_path(experiment_path)
    experiment.observers.append(FileStorageObserver(experiment_path))
    # experiment.observers.append(MongoObserver(url='mongodb://root:example@localhost', db_name=exp_name))

    # This avoids logging every tqdm progress bar update to the cout.txt file
    experiment.captured_out_filter = apply_backspaces_and_linefeeds

    return experiment


def make_experiment_tempfile(filename: str, run: Run, mode: str = 'w+b'):
    """
    Create temporary file and add it as an artifact to the run as soon as it is closed.
    """

    # We want to keep the file extension if it is specified
    _, ext = os.path.splitext(filename)

    # Windows won't let us read a tempfile while it is open, so we have to set delete=False,
    # close it and then delete it manually
    tmpfile = tempfile.NamedTemporaryFile(delete=False, mode=mode, suffix=ext)
    close_method = tmpfile.close

    def close(self):
        close_method()
        run.add_artifact(self.name, name=filename)
        # Cleanup
        try:
            os.remove(self.name)
        except FileNotFoundError:
            # We don't care if the file was already deleted
            pass

    tmpfile.close = types.MethodType(close, tmpfile)
    return tmpfile


def get_path_from_filestorage(filename: str, run: sacred.run.Run):
    for observer in run.observers:
        if isinstance(observer, FileStorageObserver):
            return os.path.abspath(os.path.join(observer.basedir, str(run._id), filename))

    raise FileNotFoundError('No FileStorageObserver attached to this run!')


def get_file_info_from_mongodb(filename: str, run: sacred.run.Run):
    for observer in run.observers:
        if isinstance(observer, MongoObserver):
            entry = observer.runs.find_one({'_id': run._id})
            database = observer.runs.database
            client = database.client
            for artifact in entry['artifacts']:
                if artifact['name'] == filename:
                    return dict(
                        host=next(iter(client.topology_description.server_descriptions())),
                        credentials=client._MongoClient__options.credentials,
                        db_name=database.name,
                        file_id=artifact['file_id']
                    )

            raise FileNotFoundError(f'No artifact with the name {filename} was found in the database!')

    raise FileNotFoundError('No MongoObserver attached to this run!')


def load_file_from_file_info(info: Dict[str, Any]):
    host, port = info['host']
    client = pymongo.MongoClient(host, port, username=info['credentials'].username,
                                 password=info['credentials'].password)
    database = client[info['db_name']]
    fs = gridfs.GridFS(database)
    return fs.get(info['file_id'])


def open_artifact_from_run(filename: str, run: sacred.run.Run, mode: str = 'r'):
    for observer in run.observers:
        if isinstance(observer, FileStorageObserver):
            path = get_path_from_filestorage(filename, run)
            return open(path, mode=mode)
        elif isinstance(observer, MongoObserver):
            info = get_file_info_from_mongodb(filename, run)
            return observer.fs.get(info['file_id'])

    raise NotImplementedError('Retrieval is only supported for FileStorageObserver and MongoObserver!')


class SerializationGuard:
    def __init__(self, obj: Any):
        self.object = obj

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        pass


def serialization_guard(*guarded_args: str) -> Callable:
    """
    Wrap the function's result in a SerializationGuard to prevent scared from writing it to the database or a JSON file
    """
    def inner_wrapper(function: Callable):
        @functools.wraps(function)
        def wrapped_fn(*args, **kwargs):
            new_args = list(args)
            for i, arg in enumerate(args):
                if isinstance(arg, SerializationGuard):
                    new_args[i] = arg.object

            new_kwargs = kwargs.copy()
            for arg_name, arg in kwargs.items():
                if isinstance(arg, SerializationGuard):
                    new_kwargs[arg_name] = arg.object

            return SerializationGuard(function(*new_args, **new_kwargs))

        wrapped_fn.guarded_args = guarded_args
        return wrapped_fn

    if len(guarded_args) == 1 and not isinstance(guarded_args, str):
        # The decorator was used without specifying any arguments. guarded_args[0] is the function to be wrapped instead
        # noinspection PyTypeChecker
        return inner_wrapper(guarded_args[0])

    return inner_wrapper


def run_command(experiment: Experiment, command: Optional[str] = None, config_updates: Optional[Dict[str, Any]] = None,
                options: Optional[Dict[str, Any]] = None) -> Run:
    """
    Run a command and wrap any parameters that are specified as "guarded" by the command in a SerializationGuard.
    If the experiment's result is wrapped in a SerializationGuard, it will be unwrapped

    :param experiment:
    :param command:
    :param config_updates:
    :param options:
    :return:
    """
    command_obj = experiment.commands[command or experiment.default_command]

    if hasattr(command_obj, 'guarded_args'):
        guarded_args = command_obj.guarded_args
    else:
        guarded_args = []

    if config_updates is None:
        new_updates = {}
    else:
        new_updates = config_updates.copy()
        for arg_name, arg in config_updates.items():
            if arg_name in guarded_args:
                new_updates[arg_name] = SerializationGuard(arg)

    run = experiment.run(command, config_updates=new_updates, options=options)
    if isinstance(run.result, SerializationGuard):
        run.result = run.result.object

    return run


def remove_sacred_garbage(obj):
    if isinstance(obj, _IDProxy):
        obj = obj.get()

    if isinstance(obj, collections.abc.Mapping):
        # Fix read only dict and list
        if 'py/object' in obj and obj['py/object'] == 'sacred.config.custom_containers.ReadOnlyDict':
            obj = dict(obj)
            obj['py/object'] = 'repair_experiments.DummyDict'
        elif 'py/object' in obj and obj['py/object'] == 'sacred.config.custom_containers.ReadOnlyList':
            obj = dict(obj)
            obj['py/object'] = 'repair_experiments.DummyList'

    if isinstance(obj, collections.abc.Mapping):
        result = dict()
        for k, v in obj.items():
            result[k] = remove_sacred_garbage(v)
    elif isinstance(obj, (list, tuple)):
        result = [remove_sacred_garbage(o) for o in obj]
    else:
        result = obj

    return result
