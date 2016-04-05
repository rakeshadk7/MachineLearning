for i in xrange(0 , len(predicted)):
    if(predicted[i] == cls[i]):
        print i, " " , predicted[i], '\t', cls[i], '\t', "MATCH"
    else:
        print i, " " , predicted[i], '\t', cls[i], '\t', "MIS-CLASSIFIED"
