import multiprocessing
import time

def func(msg):
    print('hello :', msg, time.ctime())
    time.sleep(2)
    print('end', time.ctime())
    return 'done' + msg

pool = multiprocessing.Pool(2)
result = []
for i in range(3):
    msg = 'hello %s' % i
    result.append(pool.apply_async(func=func, args=(msg,)))

pool.close()
pool.join()

for res in result:
    print('****:', res.get())             # get()函数得出每个返回结果的值

print('All end--')