import os
import pandas as pd
import numpy as np
def Get_From_Path (input_path):
    
    data =[]
    for root, dirs, files in os.walk(input_path):
        for filename in files:
            data.append(pd.read_csv(str(input_path+'/'+filename)).values)

    Df = np.concatenate(tuple(data),axis=0)
    Data = Df[:,1:26]
    Target = Df[:,26]

    return Data,Target

#D,t = Get_From_Path('ML Feature Extraction\Train\VGG16')