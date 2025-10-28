import time
import json

def loglog(level, msg, stack_trace='None'):
    info = {'msg': msg,
            'level': level,
            'stack_trace': stack_trace,
            'timestamp': time.time()}
    print(json.dumps(info))

