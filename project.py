import os.path
import numpy
import math
import scipy
from scipy.stats import mode
from scipy.stats import norm
from scipy import mean
import matplotlib.pyplot
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import time
from sklearn import datasets, linear_model, metrics 
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
import itertools
from sklearn.neighbors import KNeighborsRegressor
# NN imports
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import mean_absolute_error 
import seaborn as sb
import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
from xgboost import XGBRegressor



#####################################################
# Elizabeth Dworkin				                    #
# Some code is adapted from other online resources  #
# which are all cited in the written report.		#
#####################################################

# a function that returns the confidence interval
def confidence(data):
    c = 0.95 # we want 95 percent confidence
    n = len(data)
    alpha = 1.0 - c
    x_mean = data.mean(axis=0)
    sig = data.std()
    z_critical = scipy.stats.norm.ppf(q=0.975)
    print("z critical value = ")
    print(z_critical)
    z_interval = scipy.stats.norm.interval(alpha=c)
    stderror = sig / math.sqrt(n)
    upper = x_mean + z_critical * stderror
    lower = x_mean - z_critical * stderror
    # we are 95% sure the >>>
    print("upper value = ")
    print(upper)
    print("lower = ")
    print(lower)

## some of the below functions are adapted from code from homeworks in CS 4641
## load data randomly into: 70% train, 15% test, 15% validation
def get_test_train(fname,seed,datatype):
    '''
    Returns a test/train split of the data in fname shuffled with
    the given seed


    Args:
        fname:      A str/file object that points to the CSV file to load, passed to 
                    numpy.genfromtxt()
        seed:       The seed passed to numpy.random.seed(). Typically an int or long
        datatype:   The datatype to pass to genfromtxt(), usually int, float, or str


    Returns:
        train_X:    A NxD numpy array of training data (row-vectors), 80% of all data
        train_Y:    A Nx1 numpy array of class labels for the training data
        test_X:     A MxD numpy array of testing data, same format as train_X, 20% of all data
        test_Y:     A Mx1 numpy array of class labels for the testing data
    '''
    data = numpy.genfromtxt(fname,delimiter=',',dtype=datatype)
    data = numpy.nan_to_num(data)
    #print("asdakjshjdasd")
    #print(data)
    numpy.random.seed(seed)
    shuffled_idx = numpy.random.permutation(data.shape[0])
    cutoff = int(data.shape[0]*0.7)
    train_data = data[shuffled_idx[:cutoff]]
    test_data = data[shuffled_idx[cutoff:]]
    # Ensure there is no undefined values or numbers exceeding float capacity
    #train_data = numpy.nan_to_num(train_data)
    #test_data = numpy.nan_to_num(test_data)
    
    train_X = train_data[:,:-1].astype(float)
    train_Y = train_data[:,-1].reshape(-1,1)

    # Now set apart 50 % of the test data for Validation testing
    shuffled_idx = numpy.random.permutation(test_data.shape[0])
    cutoff = int(test_data.shape[0]*0.5)
    val_data = test_data[shuffled_idx[:cutoff]]
    test_data = test_data[shuffled_idx[cutoff:]]

    test_X = test_data[:,:-1].astype(float)
    test_Y = test_data[:,-1].reshape(-1,1)
    val_X = val_data[:,:-1].astype(float)
    val_Y = val_data[:,-1].reshape(-1,1)
    return train_X, train_Y, test_X, test_Y, val_X, val_Y, train_data, test_data
def load_data(path=''):
    #return get_test_train(os.path.join(path,'carbon_nanotubes.csv'),seed=1567708903,datatype=float)
    #return get_test_train(os.path.join(path,'hour.csv'),seed=1567708903,datatype=int)
    return get_test_train(os.path.join(path,'Concrete_Data.csv'),seed=1567708903,datatype=float)

############################
#    K nearest Neighbors   #
############################
class KNN(): 
    '''
    A very simple instance-based classifier.
    Finds the majority label of the k-nearest training points. Implements 3 distance functions:
     - euclid: standard euclidean distance (l2)
     - manhattan: taxi-cab, or l1 distance
     - mahalanobis: euclidean distance after centering and scaling by the inverse of
       the standard deviation of each component in the training data
    '''

    def __init__(self, train_X, train_Y):
        '''
        Args:
            train_X: NxD numpy array of training points. Should be row-vector form
            train_Y: Nx1 numpy array of class labels for training points.
        '''
        self.train_X = train_X
        self.train_Y = train_Y
        self.x_mean = train_X.mean(axis=0)
        self.x_std = train_X.std(axis=0)
        self.train_X_centered_scaled = (train_X-numpy.tile(self.x_mean,(train_X.shape[0],1)))/self.x_std

    def euclid(self,x_q):
        dists = numpy.zeros( (self.train_X.shape[0],1) )
        i = 0
        for dataPoint in self.train_X:
            dists[i,0] = scipy.spatial.distance.euclidean(dataPoint, x_q)
            i = i + 1
        return dists
    def query_single_pt(self,query_X,k,d):
        '''
        Returns the most common class label of the k-neighbors with the lowest distance as
        computed by the distance function d


        Args:
            query_X:    The query point, a row vector with the same shape as one row of the
                        training data in self.train_X
            k:          The number of neighbors to check
            d:          The distance function to use


        Returns:
            label:      The label of the most common class
        '''
        distances = d(query_X)
        sorted = numpy.argpartition(distances, k, axis=0)[:k]
        nearest = numpy.take(self.train_Y, sorted)
        label = mode(nearest,axis=0)[0][0]
        return label
    def query(self,data_X,k,d):
        '''
        A convenience method for calling query_single_pt on each point in a dataset, 
        such as those returned by get_test_train() or the various load_*() functions.
        If you change the API for any of the other methods, you'll probably have to rewrite
        this one.
        '''
        return numpy.array([self.query_single_pt(x_pt,k,d) for x_pt in data_X]).reshape(-1,1)
    def test_loss(self,max_k,d,test_X,test_Y):
        '''
        A convenience method for computing the misclasification rate for a range of k
        values.


        Args:
            max_k:  The maximum size of the neighborhood to test. Note, if you let this
                    be too large, it may take a very long time to finish. You may want
                    to add some code to print out after each iteration to see how long
                    things are taking
            d:      The distance function to use
            test_X: The data to compute the misclassification rate for
            test_Y: The correct labels for the test data


        Returns:
            loss:   A numpy array with max_k entries containing the misclassification
                    rate on the given data for each value of k. Useful for passing to
                    matplotlib plotting methods.
        '''
        loss = numpy.zeros(max_k)
        for k in range(1,max_k+1):
            loss[k-1] = (test_Y != self.query(test_X,k,d)).sum()/float(test_X.shape[0])
            print(k)
        return loss
    def train_loss(self, max_k, d):
        '''
        A convenience method which calls self.test_loss() on the training data. Same
        arguments as for test_loss.
        '''
        return self.test_loss(max_k,d,self.train_X,self.train_Y)

############################
#      Neural Network      #
############################
class NN():
    X_train = None 
    X_test= None 
    y_train = None 
    y_test= None 
    X_val= None 
    y_val= None 
    train = None
    test = None
    data = None

    def __init__(self, X_train, X_test, y_train, y_test, X_val, y_val):
        data = load_data()
        self.X_train = data[0]
        self.X_test = data[2]
        self.y_train = data[1]
        self.y_test = data[3]
        self.X_val = data[4]
        self.y_val = data[5]
        self.train = data[6]
        self.test = data[7]
        self.data = data

        #self.heatmap()
        #self.oneHotEncode()
        #print('There were {} columns before encoding categorical features'.format(combined.shape[1]))
        #combined = oneHotEncode(combined, cat_cols)
        #print('There are {} columns after encoding categorical features'.format(combined.shape[1]))

        NN_model = Sequential()
        global tic_time
        tic_time = time.time()

        # The Input Layer :
        NN_model.add(Dense(128, kernel_initializer='normal',input_dim = self.X_train.shape[1], activation='relu'))

        # The Hidden Layers :
        NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
        NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
        NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

        # The Output Layer :
        NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

        # Compile the network :
        NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
        NN_model.summary()

        checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
        checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
        callbacks_list = [checkpoint]

        temp = NN_model.fit(self.X_train, self.y_train, epochs=500, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)

        # Load wights file of the best model :
        wights_file = 'Weights-343--3.65645.hdf5' # choose the best checkpoint 
        NN_model.load_weights(wights_file) # load it
        NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

        print("NN fitted in %.3f s" % (time.time() - tic_time))

        #test NN
        predicted = NN_model.predict(self.X_test)
        MSE = metrics.mean_squared_error(self.y_test , predicted)
        print('Neural Net test MSE = ', MSE)
        # Get the mean absolute error on the validation data
        predicted = NN_model.predict(self.X_val)
        MSE = metrics.mean_squared_error(self.y_val , predicted)
        print('Neural Net validation MSE = ', MSE)
        print(confidence(predicted))
        #predictions = NN_model.predict(self.X_test)
        #self.make_submission(predictions[:,0],'submission(NN).csv')
        self.heatmap()


    """
    Compute the time elapsed
    """
    def tic():
        global tic_time
        tic_time = time.time()
        
    def toc():
        return time.time() - tic_time

    def heatmap(self):
        #X_train = X_train[]
        #X_train['Target'] = y_train
        C_mat = pd.DataFrame(self.X_train).corr()
        #print(C_mat)
        fig = matplotlib.pyplot.figure(figsize = (15,15))
        sb.heatmap(C_mat, vmax = .8, square = True)
        matplotlib.pyplot.show()

    def oneHotEncode(df):
        for col in range(len(X_train[0])):
            if( df[col].dtype == numpy.dtype('object')):
                dummies = pd.get_dummies(df[col],prefix=col)
                df = pd.concat([df,dummies],axis=1)
                #drop the encoded column
                df.drop([col],axis = 1 , inplace=True)
        return df

    def make_submission(self, prediction, sub_name):
        my_submission = pd.DataFrame(self.test)
        my_submission.to_csv('{}.csv'.format(sub_name),index=False)
        print('A submission file has been made')

############################
#  Linear Regression       #
############################
class lin_regression():  
    # modified from : https://www.geeksforgeeks.org/linear-regression-python-implementation/
    def __init__(self, X_train, X_test, y_train, y_test, X_val, y_val):
        # create linear regression object 
        reg = linear_model.LinearRegression()
        # train the model using the training sets 
        reg.fit(X_train, y_train)
        # regression coefficients 
        print('Coefficients: \n', reg.coef_)
        # variance score: 1 means perfect prediction 
        print('Variance score: {}'.format(reg.score(X_test, y_test))) 
        # plot for residual error 
        ## setting plot style 
        matplotlib.pyplot.style.use('fivethirtyeight')
        ## plotting residual errors in training data 
        matplotlib.pyplot.scatter(reg.predict(X_train), reg.predict(X_train) - y_train, color = "green", s = 10, label = 'Train data') 
        ## plotting residual errors in test data 
        matplotlib.pyplot.scatter(reg.predict(X_test), reg.predict(X_test) - y_test, color = "blue", s = 10, label = 'Test data')
        matplotlib.pyplot.scatter(reg.predict(X_val), reg.predict(X_val) - y_val, color = "red", s = 10, label = 'Validation data') 
        ## plotting line for zero residual error 
        matplotlib.pyplot.hlines(y = 0, xmin = 0, xmax = 80, linewidth = 2) 
        #matplotlib.pyplot.vlines(x = 0, ymin = -.0000000000006, ymax = .0000000000006, linewidth = 2) 
        ## plotting legend 
        matplotlib.pyplot.legend(loc = 'upper right') 
        ## plot title 
        matplotlib.pyplot.title("Residual errors") 
        ## function to show plot 
        matplotlib.pyplot.show() 
        #MSE
        predicted = reg.predict(X_test)
        MSE = metrics.mean_squared_error(y_test , predicted)
        print('LinearRegression test MSE = ', MSE)
        # Get the mean absolute error on the validation data
        predicted = reg.predict(X_val)
        MSE = metrics.mean_squared_error(y_val , predicted)
        print('LinearRegression validation MSE = ', MSE)
        confidence(predicted)
    def plot_data(self, X_train, X_test, y_train, y_test):
        matplotlib.pyplot.style.use('fivethirtyeight')
        print(len(X_train))
        print(len(y_train))
        date = numpy.ndarray(y_train.shape)
        season = numpy.ndarray(y_train.shape)
        year = numpy.ndarray(y_train.shape)
        month = numpy.ndarray(y_train.shape)
        hour = numpy.ndarray(y_train.shape)
        holiday = numpy.ndarray(y_train.shape)
        weekday = numpy.ndarray(y_train.shape)
        workingday = numpy.ndarray(y_train.shape)
        
        for t in range(len(X_train)):
            date[t] = (X_train[t][0])
            season[t] = (X_train[t][1])
            year[t] = (X_train[t][2])
            month[t] = (X_train[t][3])
            hour[t] = (X_train[t][4])
            holiday[t] = (X_train[t][5])
            weekday[t] = (X_train[t][6])
            workingday[t] = (X_train[t][7])

        matplotlib.pyplot.scatter(date, y_train, color = "green", s = 10, label = 'paramater 1')
        matplotlib.pyplot.scatter(season, y_train, color = "blue", s = 10, label = 'paramater 2')
        matplotlib.pyplot.scatter(year, y_train, color = "red", s = 10, label = 'paramater 3')
        matplotlib.pyplot.scatter(month, y_train, color = "cyan", s = 10, label = 'paramater 4')
        matplotlib.pyplot.scatter(hour, y_train, color = "magenta", s = 10, label = 'paramater 5')
        matplotlib.pyplot.scatter(holiday, y_train, color = "yellow", s = 10, label = 'paramater 6')
        matplotlib.pyplot.scatter(weekday, y_train, color = "black", s = 10, label = 'paramater 7')
        matplotlib.pyplot.scatter(workingday, y_train, color = "green", s = 10, label = 'paramater 8') 
        

        ## plotting line for zero residual error 
        matplotlib.pyplot.hlines(y = 0, xmin = 0, xmax = 100, linewidth = 10) 
        ## plotting legend 
        matplotlib.pyplot.legend(loc = 'upper right') 
        ## plot title 
        matplotlib.pyplot.title("Concrete Data") 
        ## function to show plot 
        matplotlib.pyplot.show() 

############################
#        Test code         #
############################
def main():
    data = load_data() #in this order: train_X, train_Y, test_X, test_Y
    X_train = data[0]
    X_test = data[2]
    y_train = data[1]
    y_test = data[3]
    X_val = data[4]
    y_val = data[5]
    

    #++++++++++++#
    #  test KNN  # uncomment this block to run the KNN best k value test
    #++++++++++++#
    """
    t0 = time.time()
    data = load_data()
    #data = get_test_train(os.path.join('','data1.dat'),seed=1568572210, datatype=float)
    test_KNN = KNN(data[0],data[1])
    k = 50
    start_k = 1
    print("KNN with euclid, from k range 1-50")

    matplotlib.pyplot.xlabel("k values")
    matplotlib.pyplot.ylabel("Test & train loss")
    matplotlib.pyplot.title("KNN with euclidean Distance")
    kNN_euc_loss = test_KNN.test_loss(k,test_KNN.euclid,data[2],data[3])
    temp = test_KNN.train_loss(k, test_KNN.euclid)
    matplotlib.pyplot.plot(range(1,k+1),kNN_euc_loss, label="testing loss", color="green")
    matplotlib.pyplot.plot(range(1,k+1),temp, label="training loss", color="red")
    matplotlib.pyplot.show()
    print("KNN with euclid, k range 1-50")
    t1 = time.time() - t0
    print("KNN test for best k completed in %.3f s" % t1)
    """

    #++++++++++++#
    #  test KNN  # uncomment this block to run the KNN
    #++++++++++++#
    """
    t0 = time.time()
    #kNN_euc_loss = test_KNN.test_loss(k,test_KNN.euclid,X_test,y_test)
    kn = KNeighborsRegressor(n_neighbors=7)
    kn.fit(X_train, y_train)
    t1 = time.time() - t0
    print("KNN fitted in %.3f s" % t1)
    y_pred = kn.predict(X_test)
    mse = metrics.mean_squared_error(y_test, y_pred)
    print(mse)
    confidence(y_pred)
    y_pred = kn.predict(X_val)
    mse = metrics.mean_squared_error(y_val, y_pred)
    print(mse)
    """

    #++++++++++++#
    #  test NN   # uncomment this line to duplicate the Neural net results!
    #++++++++++++#
    #n = NN(X_train, X_test, y_train, y_test, X_val, y_val)


    #General tests for linearity and other random things
    
    #print("x train size = ")
    #print(X_train)
    #print(y_train)
    #print("test len")
    #print(len(X_test))
    #lin = lin_regression(X_train, X_test, y_train, y_test, X_val, y_val)
    #lin.plot_data(X_train, X_test, y_train, y_test, X_val, y_val)
    
    
if __name__ == '__main__':
    main()

