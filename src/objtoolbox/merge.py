import copy

from objtoolbox import get_obj_dict, ext_setattr


def merge(dst, src):
    """
    For a given destination and source object this synchronizes both objects,
    by copying from the source into the destination object (in-place), while
    keeping the destination object structure unchanged.
    """

    t = type(dst)
    st = type(src)

    dst_attrs = get_obj_dict(dst)
    src_attrs = get_obj_dict(src)

    # now check how we should continue

    if dst_attrs is not None and src_attrs is not None:
        # handles general objects and dicts
        for k, v in dst_attrs.items():
            try:
                ext_setattr(dst, k, merge(v, src_attrs[k]))
            except KeyError:
                continue
        return dst
    elif (t == list or t == tuple) and (st == list or st == tuple):
        # handles lists and tuples
        objs = []
        for i in range(max(len(dst), len(src))):
            if i < len(src):
                sv = src[i]
                if i < len(dst):
                    # load into existing object
                    o = merge(dst[i], sv)
                else:
                    # otherwise use src object
                    o = copy.deepcopy(sv)
                objs.append(o)
            else:
                objs.append(dst[i])
        return t(objs)
    elif dst is None:
        # we have nothing to match, accept src
        return copy.deepcopy(src)
    elif dst_attrs is None:
        # this usually happens for primitive types
        if t == st:
            return copy.deepcopy(src)
        else:
            # if the types do no match, we try casting
            try:
                return t(src)
            except Exception:
                pass

    # nothing more we can do, leave unchanged
    return dst
