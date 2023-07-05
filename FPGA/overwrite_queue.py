"""
Overwrite queue with size 1

Authors
- Simon Walser
- Nicolas Tobler

"""

import threading

class OverwriteQueue:
    """
    Class to allow a single element pipeline between producer and consumer.
    """
    def __init__(self):
        self.d = None
        self.var_access_lock = threading.Lock()
        self.consumer_lock = threading.Lock()
        self.consumer_lock.acquire()

    def get(self, blocking=True, timeout=-1):
        self.consumer_lock.acquire(blocking=blocking, timeout=timeout)
        with self.var_access_lock:
            d = self.d
        return d

    def put(self, d):
        with self.var_access_lock:
            self.d = d
        if self.consumer_lock.locked():
            self.consumer_lock.release()

def _test():

    import time

    q = OverwriteQueue()

    def func():
        while True:
            time.sleep(0.01)
            i = q.get()
            print(f"get {i}")
    t = threading.Thread(target=func)
    t.daemon = True
    t.start()

    time.sleep(0.5)

    def func():
        i = 0
        while True:
            time.sleep(0.1)
            print(f"put {i}")
            q.put(i)
            i+=1
    t = threading.Thread(target=func)
    t.daemon = True
    t.start()


    time.sleep(5)


if __name__ == "__main__":
    _test()
