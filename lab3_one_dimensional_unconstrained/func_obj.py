
class FuncObj:
    def __init__(self, func, do_call_count=True):
        self.__func = func
        self.__call_count = 0
        self.__do_call_count = do_call_count

    def __call__(self, x):
        if self.__do_call_count:
            self.__call_count += 1
        return self.__func(x)

    @property
    def call_count(self):
        return self.__call_count

    def zero_call_count(self):
        self.__call_count = 0

    def set_call_count_mode(self, do_call_count):
        self.__do_call_count = self.__do_call_count
