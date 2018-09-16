import matplotlib.pyplot as plt 
from sklearn import datasets

digits = datasets.load_digits()

for label,img in zip(digits.target[:10],digits.images[:10]):
    plt.subplot(2,5,label+1) #表示域を表示
    plt.axis('off')          #枠線をoff
    plt.imshow(img,cmap=plt.cm.gray_r,interpolation='nearest')#img表示
    plt.title('{0}'.format(label)) #label表示
plt.show()    
