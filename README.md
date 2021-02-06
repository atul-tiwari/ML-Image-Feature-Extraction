# ML-Image-Feature-Extraction

```
FeatureExtraction Through Transfer Learning
```
```
Algorithms
Atul Tiwari 11800176
Atult 2208 @gmail.com,+ 919915065571
```
**Abstract :-** In this project we have used transfer learning models in

Tensorflowtoextractfeaturesofanimagetoclassifythemwiththehelp

oftraditionalmachinelearningalgorithmslikeSvm,KNNetc.Thiswill

help understand the use of transfer learning for different kind of

purposes. Then we perform multi-class classification on fruit 360 data

set to evaluate our results and success rate of our models based upon

different parameters we are also going to compare thetime to train the

algorithms.

**1. Introduction**

Deep convolutional neural network models may take days or even

weekstotrainonverylargedatasets.

In this experiment we are going to use various pre-trained models and

compare them on different parameters because In practice, very few

people train an entire Convolutional Network from scratch (with

random initialization),because it is relatively rare to have a data-set of

sufficient size. Instead, it is common to pretrain a ConvNet on a very

large data-set (e.g. ImageNet, which contains 1. 2 million images with

1000 categories),andthenuse theConvNeteitheras aninitializationor

afixedfeatureextractorforthetaskofinterest.

**1. 1 Transferlearning**

 Transferlearning involvesusing modelstrainedon oneproblem asa

```
startingpointonarelatedproblem.
```
 Transfer learning is flexible, allowing the use of pre-trained models

```
directly, as feature extraction preprocessing, and integrated into
entirelynewmodels.
```
 Kerasprovidesconvenientaccesstomanytopperformingmodelson

```
the ImageNet image recognition tasks such as VGG, Inception, and
ResNet.
```

**1. 2 Model’s**

Inthisexperimentwearegoingtouse 5 featureextractionmodelsand 5

traditionalmachinelearningmodelstoevaluateourresults

**Feature Extraction models**

**(Keras)**

```
MLAlgorithms(SK-learn)
```
## VGG 16 SVM

```
ResNet 50 V 2 KNN
MobileNetV 2 RandomForest
InceptionV 3 DecisionTree
DenseNet 121 Bagging
```
Allfeaturesextractionmodelsareusingtheirdefaultparametersandthe

onlyonespecifiedare
 Input_shape=( 100 , 100 , 3 )
 Include_top=False
 weights='imagenet'
 Training=False

The Traditional models are using following parameters which are not

mentionarealldefaultparametersprovidedbysklearn
 SVM
 kernel='rbf'
 random_state= 0
 KNN
 n_neighbors= 3
 RandomForest
 max_depth= 25
 random_state= 0
 DecisionTree
 random_state= 0
 Bagging
 base_estimator=DecisionTreeClassifier()
 n_estimators= 5
 random_state= 0


**2. DataSet**

We are using a Fruits 360 dataset: A dataset of images containing images of
variousfruitsthepropertiesofdatasetasfollows

 Totalnumberofimages: 90483.

 Trainingsetsize: 67692 images(onefruitorvegetableperimage).

 Testsetsize: 22688 images(onefruitorvegetableperimage).

 Numberofclasses: 131 (fruitsandvegetables).

 Imagesize: 100 x 100 pixels.

**2. 1 Modifications**
Theoriginaldatasetcontainstwofolderstrain andtest Ihavecombinethetwo
and place all the data in a single folder name ‘Dataset’ for ease of use in my
projectandIamusingallthe 131 classesforthisfeatureextraction.

**2. 2 Batch**
Nowafter reading all the 90483 images then we have converted them into 19
batchof 5000 eachforeaseofcomputationaftereachbatchiscompletewestore
itinacsvfileforrespectivemodelfolder

**3. TestingMethodology**

For each traditional ML model we are storingfollowingina pythonobject file
whichcanbereadthroughprintresults.pyfile

 Accuracy

 recall

 precision

 f 1 score

 Confusionmatrix

 Classificationreport
 TrainingTime( 70 %ofdata)
 TestingTime( 30 %ofdata)


**4. FeatureExtraction**

```
4. 1 VGG 16
VGG 16 isaconvolutionalneuralnetworkmodelproposedbyK.Simonyanand
A. Zisserman from the University of Oxford in the paper “Very Deep
Convolutional Networks for Large-Scale Image Recognition”. The model
achieves 92. 7 % top- 5 test accuracy inImageNet,whichisadata-setof over 14
million images belonging to 1000 classes. It was one of the famous model
submitted to ILSVRC- 2014. It makes the improvement over AlexNet by
replacing large kernel-sized filters ( 11 and 5 in the first and second
convolutionallayer,respectively)withmultiple 3 × 3 kernel-sizedfiltersoneafter
another. VGG 16 was trained for weeks and was using NVIDIA Titan Black
GPU’s.
```
VGG 16 FunctionParameters
 input_Shape=( 100 , 100 , 3 )
 Include_top=False
 weights='imagenet'

Inputlayer= 3 , 00 , 00
ModelParameters= 14714688
Output= 25

**VGG 16 Results**

**OriginalVGG 16**

```
Algorithms Accuracy Recall Precision F 1 score Traintime
(inSec)
```
```
Predtime
(inSec)
```
**SVM** 99. 78 99. 78 99. 78 99. 78 13 114

**KNN** 99. 98 99. 99 99. 98 99. 99 1 5

**Random
Forest**^99.^9499.^9399.^9499.^94812
**Decision
Tree**^90.^4690.^4690.^5490.^4771

**Bagging** 96. 34 96. 34 96. 42 96. 35 21 1


## 4. 2 RESNET 50

```
ResNet, short for Residual Networks is a classic neural network used as a
backbone for many computer vision tasks. This model was the winner of
ImageNet challenge in 2015 .Thefundamental breakthroughwithResNet wasit
allowedustotrainextremelydeepneuralnetworkswith 150 +layerssuccessfully.
Prior to ResNet training very deep neural networks was difficult due to the
problemofvanishinggradients.
ResNet- 50 that is a smaller version of ResNet 152 and frequently used as a
startingpointfortransferlearning.
```
```
ResNet 50 FunctionParameters
 input_Shape=( 100 , 100 , 3 )
 Include_top=False
 weights='imagenet'
```
```
Inputlayer= 3 , 00 , 00
ModelParameters= 23564800
Output= 25
```
```
ResNet 50 Results
```
```
OriginalResNet 50
```
```
Algorithms Accuracy Recall Precision F 1 score Tr(ianinSteicm)e P(riendSteimc)e
```
**SVM** 73. 40 73. 40 74. 19 72. 51 50 220

**KNN** 99. 52 99. 52 99. 53 99. 52 1 4

**Random
Forest**^99.^3599.^3599.^3699.^35833
**Decision
Tree**^88.^4688.^4688.^4988.^4471

**Bagging** 94. 90 94. 90 95. 07 94. 90 21 1


```
4. 3 MoblieNetV 2
```
```
Mobile-netisanfamilyofmobile-firstcomputervisionmodelsforTensorFlow,
designedtoeffectivelymaximizeaccuracywhilebeingmindful oftherestricted
resources for an on-device or embedded application. MobileNets are small,
low-latency,low-power models parameterized to meet the resource constraints
of a variety of use cases. They can be built upon for classification, detection,
embeddings andsegmentation similartohowotherpopularlarge scalemodels,
suchasInception,areused.
```
VGG 16 FunctionParameters
 input_Shape=( 100 , 100 , 3 )
 Include_top=False
 weights='imagenet'

Inputlayer= 3 , 00 , 00
ModelParameters= 2257984
Output= 25

**MobileNetV 2 Results**

**OriginalMobileNet**

```
Algorithms Accuracy Recall Precision F 1 score Tr(ianinSeticm)e P(riendSteimc)e
```
**SVM** 91. 04 91. 04 91. 11 91. 07 50 176

**KNN** 97. 45 97. 45 97. 52 97. 46 1 61

**Random
Forest**^90.^9890.^9891.^1790.^94904
**Decision
Tree**^52.^1852.^1852.^3552.^1771

**Bagging** 63. 72 63. 72 67. 68 64. 52 22 1


```
4. 4 InceptionV 3
```
```
Inception net achieved a milestone in CNN classifiers when previous models
were just going deeper to improve the performance and accuracy but
compromisingthecomputationalcost.TheInceptionnetwork,ontheotherhand,
isheavilyengineered.Itusesalotoftrickstopushperformance,bothintermsof
speed and accuracy. It is the winner of the ImageNet Large Scale Visual
RecognitionCompetitionin 2014 ,animageclassificationcompetition,whichhas
a significant improvement over ZFNet (The winner in 2013 ), AlexNet (The
winnerin 2012 )andhasrelativelylowererrorratecomparedwiththeVGGNet.
```
```
ResNet 50 FunctionParameters
 input_Shape=( 100 , 100 , 3 )
 Include_top=False
 weights='imagenet'
```
```
Inputlayer= 3 , 00 , 00
ModelParameters= 21802784
Output= 25
```
```
InceptioV 3 Results
```
```
OriginalInceptioV 3
```
```
Algorithms Accuracy Recall Precision F 1 score Tr(ianinSteicm)e P(riendSteimc)e
```
**SVM** 87. 56 87. 56 87. 99 87. 50 33 190

**KNN** 99. 50 99. 90 99. 50 99. 50 1 14

**Random
Forest**^98.^2298.^2298.^2698.^21904
**Decision
Tree**^74.^4474.^4474.^5174.^4581

**Bagging** 85. 01 85. 01 86. 13 85. 17 22 1


```
4. 5 DenseNet 121
```
```
In DenseNet, eachlayer obtains additional inputsfrom all precedinglayersand
passes on its ownfeature-maps to all subsequent layers. Concatenationisused.
Eachlayerisreceivinga“collectiveknowledge”fromallprecedinglayers.
Imageforpost
Sinceeachlayerreceivesfeaturemapsfromallprecedinglayers,networkcanbe
thinnerandcompact,i.e.numberofchannelscanbefewer.Thegrowthratekis
theadditionalnumberofchannelsforeachlayer.
So,ithavehighercomputationalefficiencyandmemoryefficiency..
```
```
ResNet 50 FunctionParameters
 input_Shape=( 100 , 100 , 3 )
 Include_top=False
 weights='imagenet'
```
```
Inputlayer= 3 , 00 , 00
ModelParameters= 7037504
Output= 25
```
```
DenseNet 121 Results
```
```
OriginalDenseNet 121
```
```
Algorithms Accuracy Recall Precision F 1 score Tr(ianinSteicm)e P(riendSteimc)e
```
**SVM** 97. 85 97. 85 97. 90 97. 84 15 146

**KNN** 99. 92 99. 92 99. 92 99. 92 1 6

**Random
Forest**^99.^7299.^7299.^7299.^72833
**Decision
Tree**^89.^4789.^4789.^5489.^4871

**Bagging** 95. 55 95. 55 95. 61 95. 54 21 1


**5. Result**

thefollowingarethecompressionbetweenthedifferentmodelsfordifferentalgorithms

```
1. SVM thehighestisthe
VGG 16
```
```
3 .DecisionTree thisistheworst
performingalgorithmoverall
```
```
4. Bagging thisisalsonot
mostreliable
```
```
2 .KNN theoverallperformance
ofthisisthebestamongall
algorithms
```
```
5 .RandomForestisthesecond
bestandfastestalgorithm
amongthese
```

**5. 1 CompressionbetweentheAccuracy**

 theabovegraphshowstheaheatmapbetweenthealgorithmsandModels
 BestmodelisVGG 16 followedbyDensenet
 BestAlgorithmisKNNfollowedbyRandomforest

**5. 2 CompressionbetweentheRunningTime**


 Theabovetableshowsthetimetakenbythemodelinonebatch+algorithm
trainingtime+PredictionTime
 RandomforestperformsbestinAlgorithms
 VGG 16 ismosttimeconsumingprocesstoExtractfeaturesfromimages

**5. 3 TimeTakenbyFeatureextractionModels**

Note:-Timeinsecond

**6. Conclusion-**

 IfiwanttogoforthemostaccuratemodelthecombinationofVGG 16 andKNN
worksbest.
 IfyouwanttogoforfastestmodelthecombinationofInceptionv 3 andrandom
ForestWork'sbest.
 IfYouwanttogoforbestoverallmodelIwouldsuggestcombinationofKNNand
InceptionV 3 .asitismoreaccuratethenthefastestcombinationandfasterthen
mostaccurateone.
 Throughalgorithmsandallmodeltheyallarehavemoreprecisionthenrecall.

DataSetLink:- https://www.kaggle.com/moltean/fruits
GitRepoLink:-https://github.com/atul-tiwari/ML-Image-Feature-Extraction

Data VGG 16 ResNet 50 MobileNet Inception DenseNet 121
SingleBatch
( 5000 )^11853152152
WholeDataset
( 90483 )^2142959272381942


