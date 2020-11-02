import numpy as np
import tensorflow as tf
import pandas as pd
import time

class Feature_extraction_VGG16:

    def save_batch_data(self,dataframe,model_name,batch_no):
        
        path = str(self.out_dir+ model_name + '/' + 'Batch ' + str(batch_no))

        dataframe.to_csv(path)

        print(path,'Done')

    def extract_features_VGG16(self,data,lables,batch_no):
        
        pre_image_data=tf.keras.applications.vgg16.preprocess_input(data)

        print(batch_no," batch Satrted")

        for model_name in self.models.keys():

            start = time.time()

            model = self.models[model_name]

            pred=model.predict(pre_image_data)
        
            features = pd.DataFrame(pred)
        
            features['target'] = lables

            self.save_batch_data(features,model_name,batch_no)

            print ("model: "+model_name + str(time.time() - start))


    def extract_data(self):

        Batch_no=0
        for image_batch,label_batch in self.data:
            Batch_no+=1
            self.extract_features_VGG16(image_batch,label_batch,Batch_no)


    def Fin_model(self,model):
        inputs=tf.keras.Input(shape=(100,100,3))

        x=model(inputs,training=False)

        x=tf.keras.layers.GlobalAveragePooling2D()(x)

        outputs=tf.keras.layers.Dense(25)(x)

        temp = tf.keras.Model(inputs,outputs)

        #print(temp.summary())

        return temp


    def init_models(self):

        input = (100,100,3)
        top = False
        weights = 'imagenet'
        
        model=tf.keras.applications.VGG16(input_shape=input,include_top=top,weights=weights)
        self.models['VGG16'] = self.Fin_model(model)
        
        model=tf.keras.applications.ResNet50V2(input_shape=input,include_top=top,weights=weights)
        self.models['ResNet50V2'] = self.Fin_model(model)
        
        model=tf.keras.applications.MobileNetV2(input_shape=input,include_top=top,weights=weights)
        self.models['MobileNetV2'] = self.Fin_model(model)
        
        model=tf.keras.applications.InceptionV3(input_shape=input,include_top=top,weights=weights)
        self.models['InceptionV3'] = self.Fin_model(model)
        
        model=tf.keras.applications.DenseNet121(input_shape=input,include_top=top,weights=weights)
        self.models['DenseNet121'] = self.Fin_model(model)

    def __init__(self,input ,output,batch_size = 5000):
    
        self.out_dir = output

        self.data=tf.keras.preprocessing.image_dataset_from_directory(input,batch_size=batch_size,image_size=(100 ,100),shuffle=False)

        self.class_names=self.data.class_names
        
        self.models = {}

        self.init_models()

        self.extract_data();



A = Feature_extraction_VGG16('fruits-360/DataSet','Train/',5000)