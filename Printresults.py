import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd

VG = open("Results/VGG16",'rb')
Re = open("Results/ResNet50V2",'rb')
Mo = open("Results/MobileNetV2",'rb')
In = open("Results/InceptionV3",'rb')
De = open("Results/DenseNet121",'rb')


VGG16 = dict(pickle.load(VG))
ResNet50V2 = dict(pickle.load(Re))
MobileNetV2 = dict(pickle.load(Mo))
InceptionV3 = dict(pickle.load(In))
DenseNet121 = dict(pickle.load(De))

models = [VGG16 ,ResNet50V2 ,MobileNetV2 ,InceptionV3,DenseNet121]
labels = ['VGG16' ,'ResNet50','MobileNet','Inception' ,'DenseNet']

################################# pt1 to print details just give in temp ################

temp = VGG16
for i in temp.keys():
    print("ACC",i,temp[i]['Accuracy'])
    print("rec",i,temp[i]['recall'])
    print("prc",i,temp[i]['precision'])
    print("f1",i,temp[i]['f1_score'])
    #print("confusion matrix",i,temp[i]['confusion_matrix'])
    #print("classification report",i,temp[i]['classification_report'])
    
    print()


############################## pt2  to plot accurcy grahs for each algo ###################



for algo in VGG16.keys():
    acc = []

    for i in models:
        acc.append(int(i[algo]['Accuracy']*100))

    x = np.arange(len(labels))  # the label locations
    width = 0.60  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, acc, width, label='Accuracy')


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage %')
    ax.set_title(algo)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    autolabel(rects1)

    fig.tight_layout()

    plt.show()

################################## pt3 to create a heat map for model vs algo ################

arr=[]
for i in models:
    temp =[]
    for j in i.keys():
        temp.append(round(i[j]['Accuracy']*100,2))
    arr.append(temp)

print(arr)

df_cm = pd.DataFrame(arr,index=labels,columns=VGG16.keys())
plt.figure(figsize = (131,131))
plt.title('Accuracy')
sn.heatmap(df_cm, annot=True,cmap="YlGnBu",fmt='g')
plt.show()

################################## pt4 to create a heat map for model vs time ################

algo_avg_time = [201, 19, 88, 8, 23]

models_time  = [118 , 53 , 15 , 21, 52] 

arr=[]

for i in algo_avg_time:
    temp = []
    for j in models_time:
        temp.append(j+i)
    arr.append(temp)

print(arr)

df_cm = pd.DataFrame(arr,index=labels,columns=VGG16.keys())
plt.figure(figsize = (131,131))
plt.title('Time Per batch + Algo Running Time')
sn.heatmap(df_cm, annot=True,cmap="viridis", fmt='g',cbar_kws={'label': 'Time in sec'})
plt.show()
