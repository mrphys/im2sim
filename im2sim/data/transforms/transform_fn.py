def normalise(x, max, min):
    if min==None:
        min = x.min()
    if max==None:
        max = x.max()
    return (x - min)/(max-min)

def standardise(x, mean=None, std=None):
    if mean==None:
        mean=x.mean()
    if std==None:
        std=x.std()
    return (x-mean)/std