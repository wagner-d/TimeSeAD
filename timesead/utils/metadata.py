"""
Collection of metadata relevant to the entire project.
"""
import logging
import os
from distutils.util import strtobool


def read_env_file(filename):
    result_dict = {}
    try:
        with open(filename, encoding='utf8', mode='r') as f:
            for line in f.readlines():
                line = line.strip()
                if line.startswith('#') or line.startswith('//'):
                    # Ignore comments
                    continue

                result = line.split('=', 1)
                if len(result) < 2:
                    logging.warning('Ignoring malformed .env line', line)
                    continue

                key, value = result
                key = key.strip()
                value = value.strip()

                # Remove string delimiters in the value that might be added
                value = value.replace('"', '')
                value = value.replace("'", '')

                result_dict[key] = value
    except (IOError, OSError) as e:
        logging.info(f'Could not read {filename}, reason: {e}. Skipping...')
        # We don't care if we can't read the file or if it does not exist
        pass

    return result_dict


def update_dict(thedict, updates):
    for k in thedict.keys():
        try:
            value = updates[k]

            # Try to convert
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    try:
                        value = bool(strtobool(value))
                    except ValueError:
                        pass

            thedict[k] = value
        except KeyError:
            # We don't care if the variable is not set
            pass


# noinspection PyUnresolvedReferences
__all__ = ['PROJECT_ROOT', 'DATA_DIRECTORY', 'LOG_DIRECTORY', 'RESOURCE_DIRECTORY', 'DISABLE_NVIDIA_SMI']

# Default configuration
settings = {}
settings['PROJECT_ROOT'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')
settings['DATA_DIRECTORY'] = os.path.join(settings['PROJECT_ROOT'], 'data')
settings['LOG_DIRECTORY'] = os.path.join(settings['PROJECT_ROOT'], 'log')
settings['RESOURCE_DIRECTORY'] = os.path.join(settings['PROJECT_ROOT'], 'resources')
settings['DISABLE_NVIDIA_SMI'] = False

# First read the .env file
real_project_root = settings['PROJECT_ROOT']
update_dict(settings, read_env_file(os.path.join(real_project_root, '.env')))

# Then read .env.local
update_dict(settings, read_env_file(os.path.join(real_project_root, '.env.local')))

# Finally use environment variables
update_dict(settings, os.environ)

# Update globals and clean up
globals().update(settings)
del settings
del real_project_root
