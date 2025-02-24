"""
A very rudimentary and opinionated storage system.

The idea is not to store and load the entire object hierarchy. Instead an
already instantiated object hierarchy is update by the stored data. This has
the advantage that we can update most of the code without thinking too much
about how this will fit the already stored data.

The storage system also takes care of handling larger numpy arrays, which makes
working with images or points clouds much easier.
"""

import os
import json
import types
import shutil
import numbers
import traceback
import fcntl

# i still like this
from pydoc import locate

import numpy as np

from objtoolbox import ext_setattr, get_obj_dict, fqn_type_name


class Skip:
    """
    This signals that a value should not be serialized (only used internally).
    """
    pass


class Serializer:
    """
    Converts an object tree into a json serializeable object tree.
    Large numpy arrays are automatically referenced and stored externally.
    """

    def __init__(self,
                 path,
                 hide_private=True,
                 mmap_arrays=False,
                 externalize=True):

        self.hide_private = hide_private
        self.mmap_arrays = mmap_arrays
        self.externalize = externalize

        if self.externalize:
            self.ext_path = os.path.join(path, "extern")
            os.makedirs(self.ext_path, exist_ok=True)

        self.used_arrays = set()

        self.obj_path = []

    def serialize(self, obj, key="", parent=None):

        if type(key) == str:
            if self.hide_private and len(key) > 0 and key[0] == "_":
                return Skip

        # skip any kind of function
        if (isinstance(obj, types.FunctionType)
                or isinstance(obj, types.MethodType)
                or isinstance(obj, types.LambdaType)):
            return Skip

        # special treatment for numpy arrays
        if type(obj) == np.ndarray:
            if obj.size > 25 and self.externalize:

                path = ".".join([str(p) for p in self.obj_path])
                file_path = os.path.join(self.ext_path, path) + ".npy"

                np.save(file_path, obj)

                if self.mmap_arrays:
                    obj = np.load(file_path, "r+")
                    if type(key) == str:
                        ext_setattr(parent, key, obj)
                    elif type(key) == int:
                        parent[key] = obj

                self.used_arrays.add(path + ".npy")
                return {
                    "__class__": "__extern__",
                    "path": path
                }
            else:
                return {
                    "__class__": fqn_type_name(obj),
                    "dtype": obj.dtype.name,
                    "data": obj.tolist()
                }

        # memory mapped arrays
        if type(obj) == np.memmap:
            if self.externalize:
                path = ".".join([str(p) for p in self.obj_path])
                file_path = os.path.join(self.ext_path, path) + ".npy"

                # only save if the memory mapping points to a different file
                if (os.path.abspath(os.path.realpath(file_path))
                    != os.path.abspath(os.path.realpath(obj.filename))):

                    np.save(file_path, obj)

                    if self.mmap_arrays:
                        obj = np.load(file_path, "r+")
                        if type(key) == str:
                            ext_setattr(parent, key, obj)
                        elif type(key) == int:
                            parent[key] = obj

                self.used_arrays.add(path + ".npy")
                return {
                    "__class__": "__extern__",
                    "path": path
                }
            else:
                np_obj = np.array(obj)
                return {
                    "__class__": fqn_type_name(np_obj),
                    "dtype": np_obj.dtype.name,
                    "data": np_obj.tolist()
                }

        if type(obj) == list or type(obj) == tuple:
            jvs = []
            for i, v in enumerate(obj):
                self.obj_path.append(i)
                jv = self.serialize(v, i, parent=obj)
                self.obj_path.pop()
                if jv is not Skip:
                    jvs.append(jv)
            return jvs

        # try to get a dict representation of the object

        attrs = None

        if isinstance(obj, dict):
            attrs = obj
        elif hasattr(obj, "__savestate__"):
            attrs = obj.__savestate__()
        elif hasattr(obj, "__getstate__"):
            attrs = obj.__getstate__()
        elif hasattr(obj, "__dict__"):
            attrs = obj.__dict__
        elif hasattr(obj, "__slots__"):
            attrs = {n: getattr(obj, n) for n in obj.__slots__}

        if attrs is None:
            # in case we don't find any serializeable attributes
            # we check if we have a primitive type and return that
            if isinstance(obj, numbers.Integral):
                return int(obj)
            if isinstance(obj, numbers.Real):
                return float(obj)
            if isinstance(obj, (str, type(None))):
                return obj
            else:
                path = ".".join([str(p) for p in self.obj_path])
                print(f"Warning: cannot save object at {path} of type {fqn_type_name(obj)}")
                return Skip

        ser_attrs = {}

        for k, v in attrs.items():
            self.obj_path.append(k)
            val = self.serialize(v, k, parent=obj)
            self.obj_path.pop()
            if val is not Skip:
                ser_attrs[k] = val

        if len(ser_attrs) == 0:
            return Skip

        # store the full type so we can compare it later
        ser_attrs["__class__"] = fqn_type_name(obj)

        return ser_attrs


class Loader:
    """
    Loads an object tree from a json file.
    External numpy arrays are automatically dereferenced and mem-mapped.
    """

    def __init__(self, path, mmap_arrays=True):

        self.mmap_arrays = mmap_arrays

        self.ext_path = os.path.join(path, "extern")

        self.used_arrays = set()

    def load(self, obj, json_obj):

        t = type(obj)
        jt = type(json_obj)

        # before we do anything else we check if we
        # can convert the json obj to a numpy array

        if jt == dict and "__class__" in json_obj:

            cls = json_obj["__class__"]

            if cls == "__extern__":
                path = json_obj["path"]
                file_path = os.path.join(self.ext_path, path) + ".npy"
                try:
                    json_obj = np.load(file_path, "r+" if self.mmap_arrays else None)
                except KeyError:
                    return None
                self.used_arrays.add(path + ".npy")
                jt = np.ndarray
            elif cls == "numpy.ndarray":
                json_obj = np.array(json_obj["data"], dtype=json_obj["dtype"])
                jt = np.ndarray

        # try to get a dict representation of the object

        attrs = get_obj_dict(obj)

        # now check how we should continue

        if attrs is not None and jt == dict:

            # handles general objects and dicts

            if hasattr(obj, "__loadstate__"):
                ld = {}
                for k, v in json_obj.items():
                    nv = self.load(None, v)
                    if nv is not Skip:
                        ld[k] = nv
                if "__class__" in ld:
                    del ld["__class__"]
                obj.__loadstate__(ld)
            elif isinstance(obj, dict) and len(obj) == 0:
                # if obj is dict-like and empty,
                # we accept whatever is stored in the json file
                ld = {}
                for k, v in json_obj.items():
                    nv = self.load(None, v)
                    if nv is not Skip:
                        ld[k] = nv
                if "__class__" in ld:
                    del ld["__class__"]
                obj.update(ld)
            else:
                for k, v in attrs.items():
                    if k in json_obj:
                        nv = self.load(v, json_obj[k])
                        if nv is not Skip:
                            ext_setattr(obj, k, nv)
            return obj
        elif (t == list or t == tuple) and jt == list:
            # handles lists and tuples
            jos = []
            for i in range(max(len(obj), len(json_obj))):
                if i < len(json_obj):
                    jv = json_obj[i]
                    if i < len(obj):
                        # load into existing object
                        jo = self.load(obj[i], jv)
                    else:
                        # otherwise create new object
                        jo = self.load(None, jv)
                    if jo is not Skip:
                        jos.append(jo)
                else:
                    jos.append(obj[i])
            return t(jos)
        elif obj is None:
            # we have nothing to match
            if jt == dict and "__class__" in json_obj:
                # try constructing a new object
                cls_name = json_obj["__class__"]
                try:
                    cls = locate(cls_name)
                    # assumes default initializeable type
                    return self.load(cls(), json_obj)
                except Exception as e:
                    traceback.print_exc()
                    print(f"Warning: skipping init of unknown type {cls_name}")
                    return Skip
            elif jt == list:
                res = [self.load(None, v) for v in json_obj]
                return [v for v in res if v is not Skip]
            else:
                # just use whatever is contained in json
                return json_obj
        elif attrs is None:
            # this usually happens for primitive types
            if t == jt:
                return json_obj
            else:
                # if the types do no match,
                # there is still a chance we can cast
                try:
                    return t(json_obj)
                except Exception:
                    pass
        else:
            # nothing more we can do
            return obj


def save(obj,
         directory,
         hide_private=True,
         mmap_arrays=False,
         externalize=True):
    """
    Stores obj under a given directory.
    The directory will be created if it not already exists.
    """

    state_path = os.path.join(directory, "state.json")

    try:
        state_fd = open(state_path, "r+")
    except FileNotFoundError:
        state_fd = None

    try:
        if state_fd is not None:
            fcntl.flock(state_fd, fcntl.LOCK_EX)

        os.makedirs(directory, exist_ok=True)

        ser = Serializer(directory,
                         hide_private=hide_private,
                         mmap_arrays=mmap_arrays,
                         externalize=externalize)
        rep = ser.serialize(obj)

        if rep is Skip:
            return False

        unfinished_path = os.path.join(directory, "unfinished.json")

        with open(unfinished_path, "w+") as fd:
            json.dump(rep, fd, indent=2)

        clean_array_store(ser)

        os.rename(unfinished_path, state_path)
    finally:
        if state_fd is not None:
            fcntl.flock(state_fd, fcntl.LOCK_UN)
            state_fd.close()

    return True


def saves(obj):
    """
    Serializes obj to a string represenation.
    """

    ser = Serializer("", externalize=False)
    rep = ser.serialize(obj)

    return json.dumps(rep)


def load(obj, path, mmap_arrays=False):
    """
    Updates obj with data stored at the given path.
    """

    state_path = os.path.join(path, "state.json")

    try:
        state_fd = open(state_path, "r+")
    except FileNotFoundError:
        return False

    try:
        fcntl.flock(state_fd, fcntl.LOCK_EX)

        json_state = json.load(state_fd)

        lod = Loader(path, mmap_arrays)
        lod.load(obj, json_state)
    finally:
        fcntl.flock(state_fd, fcntl.LOCK_UN)
        state_fd.close()

    return True


def loads(obj, string):
    """
    Updates obj with data store in the given string.
    """

    json_state = json.loads(string)

    lod = Loader("", mmap_arrays=False)
    lod.load(obj, json_state)


def clean_array_store(o):
    """
    Removes unused external arrays form the array store.
    """

    on_disk = set(os.listdir(o.ext_path))
    unused = on_disk - o.used_arrays

    for k in unused:
        os.remove(os.path.abspath(os.path.join(o.ext_path, k)))
