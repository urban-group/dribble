import timeit


def timing(func):
    def wrap(*args, **kwargs):
        t0 = timeit.default_timer()
        result = func(*args, **kwargs)
        dt = timeit.default_timer() - t0
        print("call to `{}' - elapsed time (s): {}".format(
            func.__name__, dt))
        return result
    return wrap
