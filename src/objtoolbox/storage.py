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

# i still like this
from pydoc import locate

import zarr
import numpy as np

from objtoolbox import ext_setattr, fqn_type_name


def patch_zarr_indexing():
    """
    Evil, dark-magic, monkey-patching to make zarr array indexing
    behave even more like numpy indexing.
    Will hopefully become unnecessary in future zarr versions.
    """

    original_getitem = zarr.Array.__getitem__
    original_setitem = zarr.Array.__setitem__

    def new_getitem(self, selection):

        try:
            val = original_getitem(self, selection)
        except IndexError:
            val = self.oindex[selection]

        return val

    def new_setitem(self, selection, values):

        try:
            original_setitem(self, selection, values)
        except IndexError:
            self.oindex[selection] = values

    zarr.Array.__getitem__ = new_getitem
    zarr.Array.__setitem__ = new_setitem


patch_zarr_indexing()


class Skip:
    """
    This signals that a value should not be serialized (only used internally).
    """
    pass


ZARR_CHUNK_STORES = {}


def get_chunk_store(path):

    try:
        chunk_store = ZARR_CHUNK_STORES[path]
    except KeyError:
        chunk_store = zarr.DirectoryStore(path)
        ZARR_CHUNK_STORES[path] = chunk_store

    return chunk_store


class Serializer:
    """
    Converts an object tree into a json serializeable object tree.
    Large numpy arrays are automatically referenced and stored externally.
    """

    def __init__(self, path, hide_private=True, compressor=None):

        self.hide_private = hide_private
        self.compressor = compressor

        self.ext_path = os.path.join(path, "extern")
        self.array_store = zarr.open(get_chunk_store(self.ext_path))

        self.saved_arrays = set()

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
            if obj.size > 25:

                path = ".".join([str(p) for p in self.obj_path])
                obj = self.array_store.array(
                        path,
                        obj,
                        chunks=False,
                        compressor=self.compressor,
                        overwrite=True)

                if type(key) == str:
                    ext_setattr(parent, key, obj)
                elif type(key) == int:
                    parent[key] = obj

                self.saved_arrays.add(path)
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

        # already saved arrays
        if type(obj) == zarr.core.Array:

            path = ".".join([str(p) for p in self.obj_path])

            if (self.array_store.store.path != obj.store.path
                    or obj.path != path):

                obj = self.array_store.array(
                        path,
                        obj,
                        chunks=False,
                        compressor=self.compressor,
                        overwrite=True)

                if type(key) == str:
                    ext_setattr(parent, key, obj)
                elif type(key) == int:
                    parent[key] = obj

            self.saved_arrays.add(obj.path)
            return {
                "__class__": "__extern__",
                "path": obj.path
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

        if hasattr(obj, "__savestate__"):
            attrs = obj.__savestate__()
        elif hasattr(obj, "__getstate__"):
            attrs = obj.__getstate__()
        elif hasattr(obj, "__dict__"):
            attrs = obj.__dict__
        elif hasattr(obj, "__slots__"):
            attrs = {n: getattr(obj, n) for n in obj.__slots__}
        elif isinstance(obj, dict):
            attrs = obj

        if attrs is None:
            # in case we don't find any serializeable attributes
            # we check if we have a primitive type and return that
            if isinstance(obj, (numbers.Integral, numbers.Real, str, type(None))):
                return obj
            else:
                print(f"Warning: cannot save object {key} of type {fqn_type_name(obj)}")
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

    def __init__(self, path, map_arrays=True):

        self.ext_path = os.path.join(path, "extern")

        self.map_arrays = map_arrays

        self.array_store = zarr.open(get_chunk_store(self.ext_path))

        self.loaded_arrays = set()

    def load(self, obj, json_obj):

        t = type(obj)
        jt = type(json_obj)

        # before we do anything else we check if we
        # can convert the json obj to a numpy array

        if jt == dict and "__class__" in json_obj:

            cls = json_obj["__class__"]

            if cls == "__extern__":
                path = json_obj["path"]
                try:
                    json_obj = self.array_store[path]
                except KeyError:
                    return obj
                if not self.map_arrays:
                    json_obj = np.array(json_obj)

                self.loaded_arrays.add(path)
                # we are lying about this one (actually zarr.core.Array)
                # in practice it should behave (mostly) like ndarray
                jt = np.ndarray
            elif cls == "numpy.ndarray":
                json_obj = np.array(json_obj["data"], dtype=json_obj["dtype"])
                jt = np.ndarray

        # try to get a dict representation of the object

        attrs = None

        if hasattr(obj, "__dict__"):
            attrs = obj.__dict__
        elif hasattr(obj, "__slots__"):
            attrs = {n: getattr(obj, n) for n in obj.__slots__}
        elif isinstance(obj, dict):
            attrs = obj

        # now check how we should continue

        if attrs is not None and jt == dict:

            # handles general objects and dicts

            if hasattr(obj, "__loadstate__"):
                ld = {k: self.load(None, v) for k, v in json_obj.items()}
                if "__class__" in ld:
                    del ld["__class__"]
                obj.__loadstate__(ld)
            elif isinstance(obj, dict) and len(obj) == 0:
                # if obj is dict-like and empty,
                # we accept whatever is stored in the json file
                ld = {k: self.load(None, v) for k, v in json_obj.items()}
                if "__class__" in ld:
                    del ld["__class__"]
                obj.update(ld)
            else:
                for k, v in attrs.items():
                    if k in json_obj:
                        ext_setattr(obj, k, self.load(v, json_obj[k]))
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
                return [self.load(None, v) for v in json_obj]
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


def save(obj, directory, compressor=None):
    """
    Stores obj under a given directory.
    The directory will be created if it not already exists.
    """

    # write to json

    os.makedirs(directory, exist_ok=True)

    ser = Serializer(directory, compressor=compressor)
    rep = ser.serialize(obj)

    unfinished_path = os.path.join(directory, "unfinished.json")

    with open(unfinished_path, "w+") as fd:
        json.dump(rep, fd, indent=2)

    state_path = os.path.join(directory, "state.json")

    os.rename(unfinished_path, state_path)

    # remove unused external numpy arrays

    on_disk = set(ser.array_store.keys())
    unused = on_disk - ser.saved_arrays

    for k in unused:
        del ser.array_store[k]

        # because zarr does not clean up emtpy folders
        arr_path = os.path.join(ser.ext_path, k)
        if os.path.isdir(arr_path):
            shutil.rmtree(arr_path)


def load(obj, path, map_arrays=True):
    """
    Updates obj with data stored at the given path.
    """

    state_path = os.path.join(path, "state.json")

    if not os.path.exists(state_path):
        return

    with open(state_path, "r") as fd:
        json_state = json.load(fd)

    lod = Loader(path, map_arrays)
    lod.load(obj, json_state)

    # remove unused external numpy arrays

    on_disk = set(lod.array_store.keys())
    unused = on_disk - lod.loaded_arrays

    for k in unused:
        del lod.array_store[k]

        # because zarr does not clean up emtpy folders
        arr_path = os.path.join(lod.ext_path, k)
        if os.path.isdir(arr_path):
            shutil.rmtree(arr_path)
