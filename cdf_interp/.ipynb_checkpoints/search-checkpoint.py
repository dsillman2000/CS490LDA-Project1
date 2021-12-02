from refbisect import bisect_left

def pinch_predict(data, prediction_func):

    def inner(x):
        yhat = prediction_func(x)
        if yhat < 0:
            return 0
        if yhat >= len(data):
            return len(data) - 1
        return yhat
    
    return inner

def informed_bsearch(data, prediction_func):
    def inner(x):
        refs = 0
        yhat = int(prediction_func(x))
        if yhat < 0:
            yhat = 0
        if yhat >= len(data):
            yhat = len(data) - 1
        xhat = data[yhat]

        if xhat > x:
            yhat, refs = bisect_left(data, x, 0, yhat)
        elif yhat < x:
            yhat, refs = bisect_left(data, x, yhat, len(data))
            if yhat >= len(data):
                return -1, refs
        return yhat if data[yhat] == x else -1, refs
    return inner

def exponential_search(data, prediction_func):

    def inner(x):
        ly = yhat = ry = prediction_func(x)
        
        references = 0

        if data[yhat] > x:
            i = 1
            while ly > 0 and data[ly] > x:
                references += 1
                ly = yhat - i
                i *= 2
            if ly < 0:
                ly = 0
        elif data[yhat] < x:
            i = 1
            while ry < len(data) and data[ry] < x:
                references += 1
                ry = yhat + i
                i *= 2
            if ry >= len(data):
                ry = len(data) - 1
        yhat, refs = bisect_left(data, x, ly, ry)
        return yhat if data[yhat] == x else -1, references + refs + 1
    
    return inner