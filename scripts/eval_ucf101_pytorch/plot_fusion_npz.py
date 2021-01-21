import os
import numpy as np
import matplotlib.pyplot as plt

npzPath = './fusion_model_SVTVLK/ucf101_Srgb_vgg16Tflow_vgg16_fusion_result.npz'
data = np.load(npzPath)

SResultList = data['spatialresultList']
TResultList = data['temporalresultList']
FusionResultList = data['fusionresultList']
GTList = data['groundtruthList']

SAnsList = []
TAnsList = []
#avgAnsList = []
FusionAnsList = []

for i in range(len(SResultList)):
    #avgResult = (SResultList[i]+TResultList[i])/2
    SAnsList.append(np.argmax(SResultList[i]))
    TAnsList.append(np.argmax(TResultList[i]))
    FusionAnsList.append(np.argmax(FusionResultList[i]))
    #avgAnsList.append(np.argmax(avgResult))

idx2class = {}

with open ('../../datasets/ucf101_splits/classInd.txt', 'r') as f:
    for lines in f.readlines():
        classNo, className = lines.strip('\n').split(' ')
        idx2class[int(classNo)-1] = className

xlabel=[]
for i in range(len(idx2class)):
    xlabel.append(idx2class[i])


ScorrectList = [0]*len(idx2class)
TcorrectList = [0]*len(idx2class)
FcorrectList = [0]*len(idx2class)
amountList = [0]*len(idx2class)

Smatch_count = 0
Tmatch_count = 0
match_count = 0

for i in range(len(GTList)):    
    amountList[GTList[i]] = amountList[GTList[i]] + 1    
    if (SAnsList[i] == GTList[i]):
        ScorrectList[GTList[i]] = ScorrectList[GTList[i]] + 1
        Smatch_count += 1
    if (TAnsList[i] == GTList[i]):
        TcorrectList[GTList[i]] = TcorrectList[GTList[i]] + 1
        Tmatch_count += 1        
    if (FusionAnsList[i] == GTList[i]):
        match_count += 1
        FcorrectList[GTList[i]] = FcorrectList[GTList[i]] + 1
        
print("Spatial Model Accuracy is %4.4f" % (float(Smatch_count)/len(GTList)))
print("Temporal Model Accuracy is %4.4f" % (float(Tmatch_count)/len(GTList)))
print("Fusion Accuracy is %4.4f" % (float(match_count)/len(GTList)))

SaccList = np.array(ScorrectList)/np.array(amountList)
TaccList = np.array(TcorrectList)/np.array(amountList)
FaccList = np.array(FcorrectList)/np.array(amountList)

d_FScorrectList = (np.array(FcorrectList)-np.array(ScorrectList))/np.array(amountList)
d_FTcorrectList = (np.array(FcorrectList)-np.array(TcorrectList))/np.array(amountList)

# d_FSaccList = (FaccList-SaccList)/SaccList
# d_FSaccList[np.where(SaccList==0)[0]] = 0   #remove /0 -> inf
# d_FTaccList = (FaccList-TaccList)/TaccList
# d_FTaccList[np.where(TaccList==0)[0]] = 0   #remove /0 -> inf


#### Fusion Acc report ####
        
topk = 5
worst_topk_idx = FaccList.argsort()[0:topk]
best_topk_idx = FaccList.argsort()[::-1][0:topk]
print('------ Fusion Model -----')
print('Best top {} case:'.format(topk))
for i in range(topk):
    print('Class {:<3}: {:<20} Accuracy = {} / {} = {:3.2f}%'.format(best_topk_idx[i], idx2class[best_topk_idx[i]], FcorrectList[best_topk_idx[i]], amountList[best_topk_idx[i]], FaccList[best_topk_idx[i]]*100))    
print('Worst top {} case:'.format(topk))
for i in range(topk):
    print('Class {:<3}: {:<20} Accuracy = {} / {} = {:3.2f}%'.format(worst_topk_idx[i], idx2class[worst_topk_idx[i]], FcorrectList[worst_topk_idx[i]], amountList[worst_topk_idx[i]], FaccList[worst_topk_idx[i]]*100))

plt.figure(figsize=(30,10))
plt.xticks(rotation=90, size=12)
plt.bar(xlabel, FaccList)
plt.savefig(os.path.abspath(npzPath+'/..')+'/Fusion_class_report.png')

#### Fusion vs Spatial Acc report ####
        
topk = 5
worst_topk_idx = d_FScorrectList.argsort()[0:topk]
best_topk_idx = d_FScorrectList.argsort()[::-1][0:topk]
print('------ After fusion, Compare with Spatial Model -----')
print('Best top {} case:'.format(topk))
for i in range(topk):
    print('Class {:<3}: {:<20} Accuracy Enhancement = ({} - {}) / {} = {:3.2f}%'.format(best_topk_idx[i], idx2class[best_topk_idx[i]], FcorrectList[best_topk_idx[i]], ScorrectList[best_topk_idx[i]], amountList[best_topk_idx[i]], d_FScorrectList[best_topk_idx[i]]*100))    
print('Worst top {} case:'.format(topk))
for i in range(topk):
    print('Class {:<3}: {:<20} Accuracy Enhancement = ({} - {}) / {} = {:3.2f}%'.format(worst_topk_idx[i], idx2class[worst_topk_idx[i]], FcorrectList[worst_topk_idx[i]], ScorrectList[worst_topk_idx[i]], amountList[best_topk_idx[i]], d_FScorrectList[worst_topk_idx[i]]*100))

plt.figure(figsize=(30,10))
plt.xticks(rotation=90, size=12)
plt.bar(xlabel, d_FScorrectList)
plt.savefig(os.path.abspath(npzPath+'/..')+'/Fuson_vs_Spatial_enhancement_class_report.png')

