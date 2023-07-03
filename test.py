import objtoolbox as otb


class ObjA:

    def __init__(self):

        self.b = ObjB()


class ObjB:

    def __init__(self):

        self.alist = ["hello", "this", "is", "a", "test"]


def main():
    obj = ObjA()
    val = otb.get_value_by_path(obj, "/b/alist/0/3/")
    print(val)


if __name__ == "__main__":
    main()
