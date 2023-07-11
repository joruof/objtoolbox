class bundle(dict):
    """
    A dict, which allows dot notation access.
    """

    def __init__(self, *args, **kwargs):

        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self


def get_value_by_path(obj, path):
    """
    This obtains a value from a given object, where the specific value
    can be adressed by a path given as either

    "/obj_list/0/sub_obj/value_a" or ["obj_list", 0, "sub_obj", "value_a"].
    """

    if type(path) == str:
        path = to_path_list(path)

    for p in path:
        res = None
        if type(p) == str:
            res = getattr(obj, p, None)
        if res is None:
            res = obj[p]
        obj = res

    return obj


def set_value_by_path(obj, path, value):
    """
    This set a value in a given object, where the specific value
    can be adressed by a path given as either

    "/obj_list/0/sub_obj/value_a" or ["obj_list", 0, "sub_obj", "value_a"].
    """

    if type(path) == str:
        path = to_path_list(path)

    if path == []:
        return

    sub = get_value_by_path(obj, path[:-1])

    p = path[-1]

    if type(p) == str:
        try:
            setattr(sub, p, value)
            return
        except AttributeError:
            pass

    sub[p] = value


def to_path_list(path):
    """
    This converts from a path string to a path list.
    """

    path = path.strip()
    pl = []

    for p in path.split("/"):
        if p == "":
            continue
        try:
            p = int(p)
        except ValueError:
            pass
        pl.append(p)

    return pl
            

def ext_setattr(obj, name, value):
    """
    Sets attributes for objects and key-value pairs for dicts.
    """

    if type(obj) == dict:
        obj[name] = value
    else:
        setattr(obj, name, value)


def get_obj_dict(obj):
    """
    Tries different ways to get a dict representation of an object.
    """
    
    try:
        return obj.__dict__
    except AttributeError:
        pass

    try:
        return {n: getattr(obj, n) for n in obj.__slots__}
    except AttributeError:
        pass

    if isinstance(obj, dict):
        return obj

    return None


def fqn_type_name(obj):
    """
    Returns the fully qualified type name of a given object.
    """

    cls = obj.__class__
    module = cls.__module__

    if module == 'builtins':
        return cls.__qualname__

    return module + '.' + cls.__qualname__
