from time import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time()
        res = func(*args, **kwargs)
        end = time()
        print(f"[DEBUG] running function take {end - start} seconds")
        return res
    return wrapper