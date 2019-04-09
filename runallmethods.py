from sklearn import svm
import svmcrossvalidate
from scipy.stats import multivariate_normal
from sklearn.covariance import empirical_covariance
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

#Read data from files into two matrices train and test
#and for labels into trainlabels and testlabels

avgSVMerror = 0
avgDTerror = 0
avgMVGerror = 0
data = ""
for k in range(0, 10, 1):
    f = open(data + "train."+str(k))
    mylist = f.readlines()
    train = []
    for i in range(0,len(mylist),1):
        l = mylist[i].split()
        for j in range(0,len(l),1):
            l[j] = float(l[j])
        train.append(l)
    f.close()

    f = open(data + "test."+str(k))
    mylist = f.readlines()
    test = []
    for i in range(0,len(mylist),1):
        l = mylist[i].split()
        for j in range(0,len(l),1):
            l[j] = float(l[j])
        test.append(l)
    f.close()

    f = open(data + "trainlabels."+str(k))
    mylist = f.readlines()
    trainlabels = []
    n = [0,0]
    for i in range(0,len(mylist),1):
        mylist[i] = float(mylist[i])
        trainlabels.append(mylist[i])
        if(trainlabels[i] == -1):
            trainlabels[i] = 0
        n[int(trainlabels[i])] += 1

    f.close()

    f = open(data + "testlabels."+str(k))
    mylist = f.readlines()
    testlabels = []
    for i in range(0,len(mylist),1):
        mylist[i] = float(mylist[i])
        testlabels.append(mylist[i])
        if(testlabels[i] == -1):
            testlabels[i] = 0
    f.close()

    rows = len(train)
    cols = len(train[0])

##### Decision tree #####
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train,trainlabels)
    prediction = clf.predict(test)
    err = 0
    for i in range(0, len(prediction), 1):
        if(prediction[i] != testlabels[i]):
            err += 1
    err = err/len(testlabels)
    print("DT Test error = ", err)
    avgDTerror += err

##### Cross-validated linear SVM #####

    [bestC,besterr] = svmcrossvalidate.getbestC(train,trainlabels)
    print("Best C = ", bestC)
    print("Best cross validation error = ", besterr)

    clf = svm.LinearSVC(C=bestC, max_iter=100000)
    clf.fit(train,trainlabels)
    prediction = clf.predict(test)
    err = 0
    for i in range(0, len(prediction), 1):
        if(prediction[i] != testlabels[i]):
            err += 1
    err = err/len(testlabels)
    print("SVM Test error = ", err)
    avgSVMerror += err

### MVG classification ####

    m0 = []
    m1 = []

    for i in range(0, cols, 1):
            m0.append(0)
    for i in range(0, cols, 1):
            m1.append(0)

    for i in range(0,rows,1):
            if (trainlabels[i]==0):
                    for j in range(0, cols, 1):
                            m0[j] += train[i][j]
            if (trainlabels[i]==1):
                    for j in range(0, cols, 1):
                            m1[j] += train[i][j]

    for j in range(0,cols,1):
            m0[j]= m0[j]/n[0]
            m1[j]= m1[j]/n[1]

## Separate X into two matrices, X0 and X1 one for each class
## to determine the covariance matrices

    X0 = [] #this is a matrix that contains rows from X with label 0
    X1 = [] #same as above except rows that have label 1

    for i in range(0, rows, 1):
        if(trainlabels[i] == 0):
            X0.append(train[i])
        else:
            X1.append(train[i])
                    
    ## Get covariance matrices for each class
    cov0 = empirical_covariance(X0)
    cov1 = empirical_covariance(X1)

    for i in range(0, len(cov0), 1):
        cov0[i,i] += .01
        cov1[i,i] += .01
        
    ## Predict test labels
    prediction = []
    for i in range(0, len(test), 1):
            x = test[i]
            p0 = multivariate_normal.pdf((x), m0, cov0)
            p1 = multivariate_normal.pdf((x), m1, cov1)
            if p0 > p1:
                    prediction.append(0)
            else:
                    prediction.append(1)

    err = 0
    for i in range(0, len(prediction), 1):
        if(prediction[i] != testlabels[i]):
            err += 1
    err = err/len(testlabels)
    print("MVG Test error = ", err)
    avgMVGerror += err

avgSVMerror /= 10
print("Avg SVM errror is ", avgSVMerror)
avgDTerror /= 10
print("Avg DT errror is ", avgDTerror)
avgMVGerror /= 10
print("Avg MVG errror is ", avgMVGerror)

