import numpy as np
import objtoolbox as otb


class ObjA:

    def __init__(self):

        self.b = ObjB()


class ObjB:

    def __init__(self):

        self.alist = ["hello", "this", "is", "a", "test"]
        self.arr = np.zeros((200, 200))


def main():
    obj = ObjA()
    obj.b.arr = np.ones((200, 200))

    val = otb.get_value_by_path(obj, "/b/alist/0/3/")
    print(val)

    otb.save(obj, "test")

    obj = ObjA()
    otb.load(obj, "test")
    print(np.array(obj.b.arr))

    so = otb.saves(obj)

    print(so)

    obj = ObjA()
    otb.loads(obj, so)

    print(obj.b.arr)


if __name__ == "__main__":
    main()
