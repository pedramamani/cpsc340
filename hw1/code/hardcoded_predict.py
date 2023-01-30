
def predict(x):  # x is a single example [longitude, latitude] data point
    if x[1] < 37:
        if x[0] < -96:
            return 1
        else:
            return 0
    else:
        if x[0] < -113:
            return 0
        else:
            return 1
