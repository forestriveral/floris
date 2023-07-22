import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class test(object):
    def __init__(self, value):
        print("init_start")
        self.value = value
        self.item = {}
        print("init_end")

    def __new__(cls, *args, **kwargs):
        print("new_start")
        return super().__new__(cls)

    def __str__(self):
        print("str_start")
        return str(self.value)

    def __len__(self):
        print("len_Start")
        return len(self.item)

    def __getattr__(self, item):
        print("getattr_start")
        return item

    def __setattr__(self, key, value):
        print("setattr_start")
        self.__dict__[key] = value

    def __getattribute__(self, item):
        print(" getattribute ===>", item)
        return super().__getattribute__(item)

    def __getitem__(self, item):
        print("getitem_start")
        return self.item

    def __setitem__(self, key, value):
        print("setitem_start")
        self.item[key] = value

    def __delitem__(self, key):
        print("delitem_start")

    def __get__(self, instance, value):
        print("get_start")
        print(value)
        return self.value

    def __set__(self, instance, value):
        print("set_start")
        print("...__set__...", instance, value)
        self.value = value

    def __del__(self):
        print("del_start")

    def __gt__(self, other):
        print("gt_start")
        if self.value > other.value:
            return True
        else:
            return False

    def __lt__(self, other):
        print("lt_start")
        if self.value < other.value:
            return True
        else:
            return False

    def __ge__(self, other):
        print("ge_start")
        if self.value < other.value:
            return True
        elif self.value > other.value:
            return False
        else:
            return True

    def __le__(self, other):
        print("le_start")
        if self.value < other.value:
            return True
        elif self.value > other.value:
            return False
        else:
            return True

    def __eq__(self, other):
        print("eq_start")
        if self.value == other.value:
            return True
        else:
            return False

    def __call__(self, *args, **kwargs):
        print("call_start")

    def __add__(self, other):
        return self.value+other.value

    def __radd__(self, other):
        return self.value+other

    def __sub__(self, other):
        return self.value-other.value

    def __mul__(self, other):
        return self.value*other.value

    def __iter__(self):
        self.key = 0
        return self

    def __next__(self):
        self.key +=1
        if self.key >= len(self.item):
            raise StopIteration
        else:
            return list(self.item.values())[self.key]


class testa(test):
        def __init__(self):
            pass


class B:
    a = test(1)


if __name__ == '__main__':
    # a = testa()
    # a = test(5)
    # b = B()
    a = test(1)
    # b = test(2)
    # a.zzzzz = 3
    # print(a.__diat__)
    # print(a.zzzzz)
    # print(a.__str__())
    # print(a.zzzzz)
    # print(a.value)
    # a.value = 2
    # print(a.value)
    print("***********************")
    # a['a']=1
    # a['b']=2
    # print(a['a'],a['b'])
    # del a['a']
    # print(a['a'])
    # b.a = 5
    # print(b.a)
    # del b.a
    # print(a>b)
    # print(a<b)
    # a>=b
    # a<=b
    # a==b
    # a()
    a.item =  {'a':'a','b':'b','c':'c'}
    for i in a:
        print(i)
    



