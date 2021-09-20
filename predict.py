from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


# splitting into two types of train and test, one type is a split over the whole data frame and the other is to dictionaries of 
# train and test sets for each chromosome
def split_train_test(df, percent = 0.75):
    train, test = train_test_split(df, train_size=percent, random_state=None, shuffle=True, stratify=None)
    train_df = {}
    test_df = {}
    chromosomes = df['chromosome'].unique()
    for crm in chromosomes:
        train_df[crm] = train.loc[train['chromosome'] == crm].drop(['chromosome'], axis=1)
        test_df[crm] = test.loc[test['chromosome'] == crm].drop(['chromosome'], axis=1)
        test.loc[test['chromosome'] == crm].drop(['chromosome'], axis=1)
    return train, test, train_df, test_df

# train model 1 means 1 training of the whole data, 2 means to train for each chromosome
def eval_predictions(data, features, target, train_model = 1,percent = 0.75,cores=-1,class_weight=None):
    tmp = data[features+['chromosome'] + [target]] # no need the whole table
    train,test,train_df,test_df = split_train_test(tmp,percent)
    predictions = {}
    actual = []  # a list of expected data, better to define here in case we shuffle the original table
    clf = RandomForestClassifier(n_jobs=cores, random_state=False, warm_start=False,class_weight=class_weight)
    if train_model == 1:
        clf.fit(train[features], train[target])
        for key in test_df.keys():
            actual.extend(list(test_df[key][target]))
            predictions[key] = clf.predict(test_df[key][features])
            
    if train_model == 2:
        for key in train_df.keys():
            clf.fit(train_df[key][features], train_df[key][target])
            predictions[key] = clf.predict(test_df[key][features])
    ## making confusion matrices
    total_predictions = []  # collect the prediction lists of method 1 (one training model to all tests) to one list
    for arr in predictions.values():
        total_predictions.extend(arr)
    CM = confusion_matrix(test[target], total_predictions)  # confusion matrix of all chromosomes
    return test[target], total_predictions, CM

# input: dataframe, clustering method name as appear at column names, range number of clusters (including both ends)
# output: dictionary of actual values, predictionsm and comfusion matrices
def analyze(df,features = ['index', 'length'] ,method = 'pam',clusterRange = [2,26],percent = 0.75, iterations=1,train_model=1,cores=-1):
    CM = {}
    actualVals, pred = {}, {}
    print('|', ' ' * (clusterRange[1]-clusterRange[0]+1), '|100%\n|', sep="", end="")
    for i in range(clusterRange[0],clusterRange[1]+1):
        max_f1_score = 0
        for iters in range(0,iterations):
            j = i if i > 9 else f'0{i}'
            header = f'cluster_{j}_{method}'
            actual,predictions,cm = eval_predictions(df, features, target=header,train_model=train_model,percent=percent)
            f1 = f1_score(actual, predictions,labels=None, pos_label=1, average='micro')
            if f1 > max_f1_score:
                actualVals[i] = actual
                pred[i] = predictions
                CM[i] = cm#confusion_matrix(actual, predictions)
                max_f1_score = f1
        print('-', end="")
    print('|')
    return actualVals,pred, CM


# trying to predict a whole chromosome based on other chromosomes
def predict2(df,method,features,clusterRange = [2,26],cores=-1):
    predictions = {}
    vals = {}
    CM = {}
    chro = df['chromosome'].unique()
    for i in range(clusterRange[0],clusterRange[1]+1):
        preds = {}
        j = i if i > 9 else f'0{i}'
        header = f'cluster_{j}_{method}'
        for chr in chro:
            train,test = df.loc[df['chromosome'] != chr],df.loc[df['chromosome'] == chr]
            clf = RandomForestClassifier(n_jobs=cores, random_state=False, warm_start=False)
            clf.fit(train[features], train[header])
            preds[chr] = clf.predict(test[features])
    ## making confusion matrices
        total_predictions = []  # collect the prediction lists of method 1 (one training model to all tests) to one list
        for arr in preds.values():
            total_predictions.extend(arr)
        predictions[i] = total_predictions
        vals[i] = df[header]
        CM[i] = confusion_matrix(df[header], total_predictions)  # confusion matrix of all chromosomes
    return vals,predictions,CM

# trying to predict a whole chromosome based on other chromosomes, works for a single target column
def predict4(df,target,features,cores=-1):
    chro = df['chromosome'].unique()
    preds = {}
    for chr in chro:
        train,test = df.loc[df['chromosome'] != chr],df.loc[df['chromosome'] == chr]
        clf = RandomForestClassifier(n_jobs=cores, random_state=False, warm_start=False)
        clf.fit(train[features], train[target])
        preds[chr] = clf.predict(test[features])
    ## making confusion matrices
    predictions = []  # collect the prediction lists of method 1 (one training model to all tests) to one list
    for arr in preds.values():
        predictions.extend(arr)
    vals = df[target]
    CM = confusion_matrix(df[target], predictions)  # confusion matrix of all chromosomes
    return vals,predictions,CM

# WARNING: Extremely slow algorithm
# trying to predict a one gene on the chromosome based on all other genes in the chromosome
# The method is known as 'Leave One Out'
def predict3(df,method,features,clusterRange = [2,26]):
    predictions = {}
    vals = {}
    CM = {}
    chro = df['chromosome'].unique()
    for i in range(clusterRange[0],clusterRange[1]+1):
        j = i if i > 9 else f'0{i}'
        header = f'cluster_{j}_{method}'
        preds = {}
        for chr in chro:
            crmDf = df.loc[df['chromosome'] == chr]
            predicted_cluster = []
            for index in crmDf.index:
                train,test = df[df.index!=index],df[df.index==index]
                clf = RandomForestClassifier(n_jobs=6, random_state=False, warm_start=False)
                clf.fit(train[features], train[header])
                predicted_cluster.append(clf.predict(test[features]))
            preds[chr] = predicted_cluster
    ## making confusion matrices
        total_predictions = [] 
        for arr in preds.values():
            total_predictions.extend(arr)
        predictions[i] = total_predictions
        vals[i] = df[header]
        CM[i] = confusion_matrix(df[header], total_predictions)  # confusion matrix of all chromosomes
    return vals,predictions,CM


# WARNING: Leave one out, single target column
def predict5(df,target,features,cores=-1):
    predictions = {}
    vals = {}
    CM = {}
    chro = df['chromosome'].unique()
    preds = {}
    for chr in chro:
        crmDf = df.loc[df['chromosome'] == chr]
        predicted_cluster = []
        for index in crmDf.index:
            train,test = df[df.index!=index],df[df.index==index]
            clf = RandomForestClassifier(n_jobs=cores, random_state=False, warm_start=False)
            clf.fit(train[features], train[target])
            predicted_cluster.append(clf.predict(test[features]))
        preds[chr] = predicted_cluster
    ## making confusion matrices
    total_predictions = [] 
    for arr in preds.values():
        total_predictions.extend(arr)
    CM = confusion_matrix(df[target], total_predictions)  # confusion matrix of all chromosomes
    return df[target],total_predictions,CM


# Plotting the prediction score graph of PAM clustering from 2 to 26 clusters
def plotit(vals, preds,title = 'predicion score',score = 'precision',avg='micro',multiplot=False,colour=None,labels=False):
    if not multiplot:
        minClusters = min(vals.keys()) ## minimum number of clusters 
        x = []
        if score == 'precision':
            for i in range(minClusters,int(len(vals)+minClusters)):
                x = x +[precision_score(vals[i], preds[i], labels=None, pos_label=1, average=avg, sample_weight=None, zero_division='warn')]
        elif score == 'f1':
            for i in range(minClusters,int(len(vals)+minClusters)):
                x = x +[f1_score(vals[i], preds[i],labels=None, pos_label=1, average=avg)]
        else: 
            print('invalid scoring method')
        if colour is None:
            plt.plot(range(minClusters, len(x)+minClusters),x)
        else:
            plt.plot(range(minClusters, len(x)+minClusters),x,color=colour)            
    if multiplot:
        minClusters = min(list(vals[list(vals.keys())[0]].keys())) ## minimum number of clusters 
        di = {}
        for i in vals.keys():
            x = []
            if score == 'precision':
                for j in range(minClusters,int(len(vals[i])+minClusters)):
                    x = x +[precision_score(vals[i][j], preds[i][j], labels=None, pos_label=1, average=avg, sample_weight=None, zero_division='warn')]
                di[i] = x
            elif score == 'f1':
                for j in range(minClusters,int(len(vals[i])+minClusters)):
                    x = x +[f1_score(vals[i][j], preds[i][j],labels=None, pos_label=1, average=avg)]
                di[i] = x
            else: print('invalid scoring method')
            if colour is None:
                if not labels:
                    plt.plot(range(minClusters, len(x)+minClusters),x)
                else:
                    plt.plot(range(minClusters, len(x)+minClusters),x,label=labels[i])
            else:
                if not labels: 
                    plt.plot(range(minClusters, len(x)+minClusters),x,color=colour)
                else:
                    plt.plot(range(minClusters, len(x)+minClusters),x,color=colour)
    plt.title(title)
    if labels:
        plt.legend()
    plt.xlabel('Number of clusters')
    plt.ylabel('precision')
    plt.grid()
    plt.show()


    
# Plot confusion matrix
def plotCM(CM,title,dpi = 500,fontSize = 20):
    plt.rcParams.update({'font.size': fontSize})
    classNames = range(1,len(CM)+1)
    disp = ConfusionMatrixDisplay(confusion_matrix=CM,display_labels=classNames)
    disp.plot(include_values=True, cmap='Blues', ax=None, xticks_rotation='horizontal')
    plt.gcf().set_dpi(dpi)
    plt.title(title)
    plt.show()