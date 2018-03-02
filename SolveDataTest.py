from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import math as m


pi = np.pi

def euclid(x,y):
    norm = np.subtract(x,y)
    return m.sqrt(norm[0]**2+norm[1]**2+norm[2]**2)

#PREPROCESSING START
#data preprocessing dari file menjadi data perkelas
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

f = open('data_test_PNN.txt','r')
bacabaris = f.readlines()
datatest = []
for i in range(1,len(bacabaris)):
        splitted = bacabaris[i].replace('\n','').replace('\r','').split('\t')
        datatest.append([float(splitted[0]),float(splitted[1]),float(splitted[2])])

kelas0 = [[],[],[],[]]
kelas1 = [[],[],[],[]]
kelas2 = [[],[],[],[]]
xkelas0 = []
xkelas1 = []
xkelas2 = []
for i in range(len(x1)):
    if (y[i] == 0):
        xkelas0.append([x1[i],x2[i],x3[i]])
    elif (y[i] == 1):
        xkelas1.append([x1[i],x2[i],x3[i]])
    elif (y[i] == 2):
        xkelas2.append([x1[i],x2[i],x3[i]])
#PREPROCESSING END
		

##FINDING SMOOTHING START
#Mencari jarak antar node untuk mencari nilai smoothing
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

print ("Input nilai g yang ingin digunakan (default = 1.4192775982427548)")
inputan = input()
try:
    if (inputan == ""):
        print ("Menggunakan default")
        g = 1.4192775982427548  #Nilai G Hasil Optimasi
    else:
        g = float(inputan)
except ValueError:
    print ('Tidak dapat di convert ke float, menggunakan default')
    g = 1.4192775982427548  #Nilai G Hasil Optimasi
    
    
    
#g = 1.4192775982427548  #Nilai G Hasil Optimasi
m = 3.0
tau0 = g * distance0_average
tau1 = g * distance1_average
tau2 = g * distance2_average

#FINDING DISTANCE END

hasil = ""
print ("Generating prediction .....")
for xtes in datatest:
    probs = []
    #MENGHITUNG PROBABILITAS PER KELAS START
	
	#Kelas 0
    hasilexp = 0
    for i in range(len(xkelas0)):
        hasilexp+=np.exp(-(np.linalg.norm(np.subtract(xtes,xkelas0[i])/(2*(tau0 ** 2)))))
    prob = hasilexp / (((2*pi) ** (m/2))*(tau0 ** m)*len(xkelas0))
    probs.append(prob)
	
	#Kelas 1
    hasilexp = 0
    for i in range(len(xkelas1)):
        hasilexp+=np.exp(-(np.linalg.norm(np.subtract(xtes,xkelas1[i])/(2*(tau1 ** 2)))))
    prob = hasilexp / (((2*pi) ** (m/2))*(tau1 ** m)*len(xkelas1))
    probs.append(prob)
	
	#Kelas 2
    hasilexp = 0
    for i in range(len(xkelas2)):
        hasilexp+=np.exp(-(np.linalg.norm(np.subtract(xtes,xkelas2[i])/(2*(tau2 ** 2)))))
    prob = hasilexp / (((2*pi) ** (m/2))*(tau2 ** m)*len(xkelas2))
    probs.append(prob)
    output = probs.index(max(probs))
    hasil += str(output) + "\n"
    print (output)
    #MENGHITUNG PROBABILITAS PER KELAS END
f = open("outputs/prediksi.txt","w")
f.write(hasil)
f.close()
print ("Prediction Saved on \"prediksi.txt\"")        
