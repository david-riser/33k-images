import importlib
import sys
import traceback


def create(cls):
    '''expects a string that can be imported as with a module.class name'''
    module_name, class_name = cls.rsplit(".",1)

    try:
        somemodule = importlib.import_module(module_name)
        cls_instance = getattr(somemodule, class_name)

    except Exception as err:
        print("Creating directories error: {0}".format(err))
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb)
        exit(-1)

    return cls_instance
