from functools import partial


class call_count(object):

    instances = {}

    def __init__(self, func):
        self.func = func
        self.num_calls = 0
        call_count.instances[func] = self

    def __call__(self, *args, **kwargs):
        self.num_calls += 1
        return self.func(*args, **kwargs)

    def __get__(self, obj, objtype):
        return partial(self, obj)

    @staticmethod
    def all_counts():
        return dict([(func.__name__, call_count.instances[func].num_calls) for func in call_count.instances])

    @staticmethod
    def zero_all_counts():
        for func in call_count.instances:
            call_count.instances[func].num_calls = 0
