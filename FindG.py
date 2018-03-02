from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

    
#cari distance
def euclid(x,y):
    norm = np.subtract(x,y)
    return np.sqrt(norm[0]**2+norm[1]**2+norm[2]**2)
    #return norm

#data preprocessing
f = open('data_train_PNN.txt','r')
bacabaris = f.readlines()
x1 = []
x2 = []
x3 = []
y = []
for i in range(1,len(bacabaris)):
        splitted = bacabaris[i].replace('\n','').replace('\r','').split('\t')
        x1.append(float(splitted[0]))
        x2.append(float(splitted[1]))
        x3.append(float(splitted[2]))
        y.append(float(splitted[3]))


kf = KFold(n_splits=5)
g = 0
daftarg = []
daftarakurasi = []
output_csv = "g,akurasi\n"
for loop in range(1000):
    g +=np.random.uniform(0,0.01)
    akurasis = []
    for train_index, test_index in kf.split(x1):
        xkelas0 = []
        xkelas1 = []
        xkelas2 = []
        xsemua = []
        ytes = []
        for i in train_index:
            if (y[i] == 0):
                xkelas0.append([x1[i],x2[i],x3[i]])
            elif (y[i] == 1):
                xkelas1.append([x1[i],x2[i],x3[i]])
            elif (y[i] == 2):
                xkelas2.append([x1[i],x2[i],x3[i]])
        for i in test_index:
            xsemua.append([x1[i],x2[i],x3[i]])
            ytes.append(y[i])
        
        distance0 = 0
        for i in range(len(xkelas0)):
            temp = []
            for j in range(len(xkelas0)):
                if (i != j):
                    jarak = euclid(xkelas0[i],xkelas0[j])
                    temp.append(jarak)
            distance0+=min(temp)
        distance0_average = distance0/len(xkelas0)
        
        distance1 = 0
        for i in range(len(xkelas1)):
            temp = []
            for j in range(len(xkelas1)):
                if (i != j):
                    jarak = euclid(xkelas1[i],xkelas1[j])
                    temp.append(jarak)
            distance1+=min(temp)
        distance1_average = distance1/len(xkelas1)
        
        distance2 = 0
        for i in range(len(xkelas2)):
            temp = []
            for j in range(len(xkelas2)):
                if (i != j):
                    jarak = euclid(xkelas2[i],xkelas2[j])
                    temp.append(jarak)
            distance2+=min(temp)
        distance2_average = distance2/len(xkelas2)
        pi = np.pi
        m = 3.0
        tau0 = g * distance0_average
        tau1 = g * distance1_average
        tau2 = g * distance2_average
        benarz = 0
        semuaz = 0
        for j in range(len(xsemua)):
            probs = []
            hasilexp = 0
            for i in range(len(xkelas0)):
                hasilexp+=np.exp(-(np.linalg.norm(np.subtract(xsemua[j],xkelas0[i])/(2*(tau0 ** 2)))))
            prob = hasilexp / (((2*pi) ** (m/2))*(tau0 ** m)*len(xkelas0))
            probs.append(prob)
    
            hasilexp = 0
            for i in range(len(xkelas1)):
                hasilexp+=np.exp(-(np.linalg.norm(np.subtract(xsemua[j],xkelas1[i])/(2*(tau1 ** 2)))))
            prob = hasilexp / (((2*pi) ** (m/2))*(tau1 ** m)*len(xkelas1))
            probs.append(prob)
    
            hasilexp = 0
            for i in range(len(xkelas2)):
                hasilexp+=np.exp(-(np.linalg.norm(np.subtract(xsemua[j],xkelas2[i])/(2*(tau2 ** 2)))))
            prob = hasilexp / (((2*pi) ** (m/2))*(tau2 ** m)*len(xkelas2))
            probs.append(prob)
            if (ytes[j] == probs.index(max(probs))):
                benarz+=1
                semuaz+=1
            else:
                semuaz+=1
        akurasi = benarz/semuaz
        akurasis.append(akurasi)
    print ("g = ",g,"akurasi =",sum(akurasis)/len(akurasis))
    daftarg.append(g)
    daftarakurasi.append(sum(akurasis)/len(akurasis))
    output_csv += str(g) + "," + str(akurasi) + "\n"
plt.plot(daftarg,daftarakurasi)
plt.xlabel("Nilai g")
plt.ylabel("Nilai Akurasi")
plt.show()
f = open("outputs/findingG.csv","w")
f.write(output_csv)
f.close()
