"""
Collection of general utility functions.
"""
import collections.abc
import functools
import importlib
import inspect
import itertools
import numpy as np
from typing import List, TypeVar, Tuple, Union, Dict, Any, Iterator, Optional

T = TypeVar('T')


def split_list(l : List[T], n : int) -> List[List[T]]:
    """Splits a list into n roughly equal sized sublists.

    :param l: Arbitrary list.
    :type l: List[T]
    :param n: Number of pieces the list is to be divided into.
    :type n: int
    :return: List of sublists of roughly equal size.
    :rtype: List[List[T]]
    """

    assert len(l) >= n

    min_piece_size, rest = divmod(len(l), n)

    return list((l[i * min_piece_size + min(i, rest):(i + 1) * min_piece_size + min(i + 1, rest)] for i in range(n)))


def generate_intervals(n: int, min_size: int, max_size: int, length: int) -> List[Tuple[int, int]]:
    """Generates n random non overlapping intervals of indices with size between min_size and max_size in range(length).

    :param n: Number of intervals.
    :type n: int
    :param min_size: Minimum size of generated intervals.
    :type min_size: int
    :param max_size: Maximum size of generated intervals.
    :type max_size: int
    :param length: Length of range to compute the intervals for.
    :type length: int
    :return: List intervals.
    :rtype: List[Tuple[int, int]
    """

    assert n * min_size <= length

    total = length - n * min_size

    # Compute n integers larger than or equal to min_size, which sum to length.
    candidates = [0] + sorted(list(np.random.randint(0, high=total + 1, size=n - 1))) + [total]
    candidates = [h - l for h, l in zip(candidates[1:], candidates[:-1])]
    candidates = [candidate + min_size for candidate in candidates]

    # Sample left boundaries of intervals
    left  = [np.random.randint(0, high=candidate - min_size + 1) for candidate in candidates]
    right = [np.random.randint(l + min_size, high=min(candidate + 1, l + max_size + 1)) for l, candidate in zip(left, candidates)]

    return [(l + sum(candidates[:idx]), r + sum(candidates[:idx])) for idx, (l, r) in enumerate(zip(left, right))]


def pack_tuple(x: Union[T, Tuple[T, ...]]) -> Tuple[T, ...]:
    if isinstance(x, tuple):
        return x

    return x,


def ceil_div(a: int, b: int) -> int:
    # assert a >= 0 and b > 0
    return (a + b - 1) // b


class Bunch:
    def __new__(cls, *args, **kwargs):
        # Need to do this in a weird way because pickle doesn't invoke __init__ when unpickling,
        # so _adict is not registered and we get an infinite loop because it is calling __getattr__
        instance = super().__new__(cls)
        instance._adict = {}
        return instance

    def __init__(self, adict):
        self._adict = adict

    def __getattr__(self, item):
        try:
            return self._adict[item]
        except KeyError:
            return super(Bunch, self).__getattribute__(item)

    def __getitem__(self, item):
        return self._adict[item]


def str2cls(fully_qualified_name: str, base_module: Optional[str] = None):
    # Ignore if it is already a class
    if inspect.isclass(fully_qualified_name):
        return fully_qualified_name

    fully_qualified_name = fully_qualified_name.rsplit('.', 1)
    if len(fully_qualified_name) == 2:
        # load the module, will raise ImportError if module cannot be loaded
        m = importlib.import_module(fully_qualified_name[0])
    elif base_module is not None:
        m = importlib.import_module(base_module)
    else:
        m = Bunch(globals())

    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, fully_qualified_name[-1])
    return c


def objspec2constructor(obj_spec: Union[str, Dict[str, Union[str, Dict[str, Any]]], Any],
                        base_module: Optional[str] = None, **kwargs) -> Any:
    """
    This method turns an object spec into a new instance of the object specified by it.

    :param obj_spec: If this is a string, it will be considered a class name that you want to instantiate or
        a function name that you want to call. The name can be either fully qualified or part of `base_module`.
        Alternatively, this can be a mapping of the form `{'class': 'MyClass', 'args': {'arg1': 'val'}}`
        or without the args `{'class': 'MyClass'}`. Then `obj_spec['class']` will be instantiated like in the string
        case, but with the additional keyword arguments specified in obj_spec['args'] if they are present.
        If obj_spec does not fall into any of those cases, this method will just return obj_spec.
    :param base_module: Name of a module where to look for class names.
    :param kwargs: Additional keyword arguments that will be passed to the constructor while instantiating the object.
        Note that these will take precedence over the values specified in `obj_spec['args']`.
    :return: An object instantiated according to obj_spec.
    """

    if isinstance(obj_spec, str):
        # We are given a class name or generally the name of a Callable
        # Try to import it
        cls = str2cls(obj_spec, base_module=base_module)
        # We don't have any args in the spec in this case, use the ones provided by the call to objspec2obj
        return functools.partial(cls, **kwargs)

    if not isinstance(obj_spec, collections.abc.Mapping):
        # obj_spec is not a valid object specification, we just assume it is already the class
        return obj_spec

    # The third case is that we got a dict of the form
    # {'class': 'MyClass', 'args': {'arg1': 'val'}} or without any args {'class': 'MyClass'}
    cls = str2cls(obj_spec['class'], base_module=base_module)
    spec_args = dict(obj_spec['args']) if 'args' in obj_spec else {}
    # kwargs input to this call take precedence over those specified in the spec
    spec_args.update(kwargs)

    # Finally we instantiate and return the object
    return functools.partial(cls, **spec_args)


def param_grid_to_list_of_dicts(param_grid: Union[Dict[str, Union[Dict, List]], List]) -> Iterator[Dict[str, Any]]:
    if not isinstance(param_grid, dict):
        # Input is a list
        for p in param_grid:
            yield p

        return

    # param_grid is a dict
    iterators = (param_grid_to_list_of_dicts(v) for v in param_grid.values())
    for res in itertools.product(*iterators):
        yield {k: v for k, v in zip(param_grid.keys(), res)}


def getitem(x, item):
    return x[item]


def halflife2alpha(halflife: float) -> float:
    return 1 - 0.5**(1 / halflife)

