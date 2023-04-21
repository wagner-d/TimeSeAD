import os
import errno


def check_path(path : str):
    '''Checks whether a directory exists and creates it if it doesn't.

    :param path: Path of directory to check.
    :type path: str
    '''

    try:
        os.makedirs(path)

    except OSError as exc:

        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass

        else:
            raise
