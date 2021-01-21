import numpy as np
import matplotlib.pyplot as plt

data = np.load('ucf101_flow_mobilenet_model_result.npz')

#data['resultList']
correctList = data['correctList']
amountList = data['amountList']

accList = correctList/amountList

idx2class = {}

with open ('../../datasets/ucf101_splits/classInd.txt', 'r') as f:
    for lines in f.readlines():
        classNo, className = lines.strip('\n').split(' ')
        idx2class[int(classNo)-1] = className
        
topk = 5
worst_topk_idx = accList.argsort()[0:topk]
best_topk_idx = accList.argsort()[::-1][0:topk]

#worst_topk_idx = worst_topk_idx + 1
#best_topk_idx = best_topk_idx + 1

print('Best top {} case:'.format(topk))
for i in range(topk):
    print('Class {:<3}: {:<20} Accuracy = {} / {} = {:3.2f}%'.format(best_topk_idx[i], idx2class[best_topk_idx[i]], correctList[best_topk_idx[i]], amountList[best_topk_idx[i]], accList[best_topk_idx[i]]*100))
    
print('Worst top {} case:'.format(topk))
for i in range(topk):
    print('Class {:<3}: {:<20} Accuracy = {} / {} = {:3.2f}%'.format(worst_topk_idx[i], idx2class[worst_topk_idx[i]], correctList[worst_topk_idx[i]], amountList[worst_topk_idx[i]], accList[worst_topk_idx[i]]*100))

xlabel=[]
for i in range(len(idx2class)):
    xlabel.append(idx2class[i])

plt.figure(figsize=(30,10))
plt.xticks(rotation=90, size=12)
plt.bar(xlabel, accList)
plt.savefig('class_report.png')
