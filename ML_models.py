import Get_data 
import pickle 
import time

class ML_models:

    def save_Scores(self,pred,m_name):
        
        import sklearn.metrics as MT
        
        curr_model = {}
        
        curr_model['Accuracy'] = MT.accuracy_score(self.Test['Labels'], pred)
        curr_model['confusion_matrix'] = MT.confusion_matrix(self.Test['Labels'], pred)
        curr_model['f1_score'] = MT.f1_score(self.Test['Labels'], pred, average='weighted')
        curr_model['classification_report'] = MT.classification_report(self.Test['Labels'], pred)
        curr_model['recall'] = MT.recall_score(self.Test['Labels'], pred,average='weighted')
        curr_model['precision'] = MT.precision_score(self.Test['Labels'], pred,average='weighted')

        self.result[m_name] = curr_model
        print("Acc = ",curr_model['Accuracy'])
        #print ("Process test time: " + str(time.time() - self.start))
        


    def compute(self):
        
        for key in self.Models.keys():
            print(key)
            model = self.Models[key]

            #self.start = time.time()

            model.fit(self.Train['Data'],self.Train['Labels'])

            #print ("Process train time: " + str(time.time() - self.start))
            #self.start = time.time()

            pred = model.predict(self.Test['Data'])

            self.save_Scores(pred,key)

    def Load_Data(self,Data_path):
        
        data,Lables = Get_data.Get_From_Path(Data_path)

        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(data, Lables, test_size=0.30, random_state=1)

        self.Train =  {}
        self.Train['Data'] = X_train
        self.Train['Labels'] = y_train

        self.Test =  {}
        self.Test['Data'] = X_test
        self.Test['Labels'] = y_test

    def init_Models(self):
        
        from sklearn.svm import SVC
        self.Models['SVM'] = SVC(kernel ='rbf', random_state =0)

        from sklearn.neighbors import KNeighborsClassifier
        self.Models['KNN']= KNeighborsClassifier(n_neighbors=3)

        from sklearn.ensemble import RandomForestClassifier
        self.Models['Random_Forest'] = RandomForestClassifier(max_depth=25, random_state=0)

        from sklearn.tree import DecisionTreeClassifier
        self.Models['Decision_Tree']= DecisionTreeClassifier(random_state=0)

        from sklearn.ensemble import BaggingClassifier
        self.Models['Bagging'] = BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=5, random_state=0)



    def __init__(self,Data_path):
        
        self.start = time.time()
        self.Models ={}
        self.init_Models()

        self.Load_Data(Data_path)

        self.result = {}
        self.compute()

        ## save result in a file
        filehandler = open('Results/'+ Data_path [Data_path.find('/')+1:], 'wb') 
        pickle.dump(self.result, filehandler)    

M1 = ML_models('Train/DenseNet121')
#M1 = ML_models('Train/VGG16')
#you can change this path for the model change and run