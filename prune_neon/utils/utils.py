def stop_func(s, v):
    if s is None:
        return tuple([v, False])
    return tuple([min(v, s), v > s])


def merge_dict(x, y):
    z = x.copy()
    z.update(y)
    return z
