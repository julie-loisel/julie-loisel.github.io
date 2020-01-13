---
title: "Statistical learning (IMC-4302C)"
permalink: IMC-4302C/Lab1/
---

# 1. Linear regression

In this practice session, you are invited to train a linear regression model using gradient descent method. After the learning phase, your model should predict house prices in the region of *Ile-de-France* given their areas (in m²) and their numbers of rooms.

We will also enhace the perfomence of the learning algorithm using different implementation techniques like vectorization and features normalization.

### Import libraries and load data
Import **numpy** library that support matrix operation and **matplotlib** library for plotting data.  
<font color="blue">**Question 1: **</font>The *"house.csv"* file contains 3 columns that represent the area, the number of rooms and the price of 600 houses (one per row). 
- Open this file with a file editor to understand more the data. 
- Load the data in "house_data" variable and check its size.  



```python
import numpy as np
import matplotlib.pyplot as plt

house_data=np.loadtxt('house.csv') 
# you could verify the size of the data using shape() function on numpy array house_data
print(house_data.shape)
print(house_data)
```

    (600, 3)
    [[122.   6. 361.]
     [125.   6. 759.]
     [ 79.   4. 584.]
     ...
     [ 86.   4. 332.]
     [ 64.   3. 246.]
     [100.   5. 249.]]


### Linear regression with 1 feature (house area)

In this first part, we will train a linear model for house price prediction using only one feature the house area. We will start by implementing a cost function and the gradient of this cost function. Then, we will implement the gradient descent algorithm that minimizes this cost function and determine the linear model parameter $\theta$ in the equation $h_\theta(x)=\theta_1 x$.  

<font color="blue">**Question 2: **</font> 
- Determine the number of samples "m" from the input data "house_price".
- Extract the house area and price columns respectively in "x_1" and "y" arrays to visualize them.  
**Hint:** The shape of "x_1" and "y" arrays should be (m,1) for the following questions and not (m,). You could use [newaxis numpy](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#basic-slicing-and-indexing) object to add a new axis of length one.


```python
%matplotlib notebook

m = house_data.shape[0] # number of sample
n = 1                   # number of features
x_1 = house_data[:,0,np.newaxis] # we add np.newaxis in the indexing to obtain an array 
print(x_1.shape)                 # with shape (600,1) instead of (600,)
X = x_1
y = house_data[:,2,np.newaxis] # we add np.newaxis in the indexing to obtain an array with shape (600,1) instead of (600,)
plt.figure("Visualize house data",figsize=(9,5))
plt.scatter(x_1, y,  color='black')
plt.xlabel('house area (m²)')
plt.ylabel('house price (1000€)')
plt.title('house area vs price')
plt.show()
```

    (600, 1)



    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA4QAAAH0CAYAAABl8+PTAAAgAElEQVR4nOzde3xU9Z3/8TOJJAFCEnLhYhKtrrbinbZaqy3KWq1VV7BWkLrloq1amEq6NCouPDBWLS7iVFdxvVWX0iCh8VJ3vabNdlu71fIQEaGK4g2vrVaQh8hI5P37g99MM8lczsyc8z2XeT0fj/PYmkyS70zyZT/v+Zzv92sJAAAAAFCSLK8HAAAAAADwBoEQAAAAAEoUgRAAAAAAShSBEAAAAABKFIEQAAAAAEoUgRAAAAAAShSBEAAAAABKFIEQAAAAAEoUgRAAAAAAShSBEAAAAABKFIEQAAAAAEoUgRAAAAAAShSBEAAAAABKFIEQAAAAAEoUgRAAAAAAShSBEAAAAABKFIEQAAAAAEoUgRAAAAAAShSBEAAAAABKFIEQAAAAAEoUgRAAAAAAShSBEAAAAABKFIEQAAAAAEoUgRAAAAAAShSBEAAAAABKFIEQAAAAAEoUgRAAAAAAShSBEAAAAABKFIEQAAAAAEoUgRAAAAAAShSBEAAAAABKFIEQAAAAAEoUgRAAAAAAShSBEAAAAABKFIEQAAAAAEoUgRAAAAAAShSBEAAAAABKFIEQAErIokWLZFmW/vrXv3o9FPjcK6+8IsuydNddd3k9FACAiwiEAFBCCISwi0AIAKWBQAgAJYRACLt2796tjz/+WH19fV4PBQDgIgIhAJQQAmF6n376qT7++GOvh+ELu3btUjwe93oYAABDCIQAUEISgfDFF1/UjBkzVFtbq5qaGs2cOVMfffRRymN37dqlK6+8Uvvvv78qKiq07777av78+dq5c2fK4yzL0qJFiwb9rH333VczZsxI/vcnn3yiK664QgcccIAqKytVX1+v4447To899ljK1/35z3/WWWedpZEjR6qyslJf+MIX9MADD9h6fkuWLNGXv/xl1dfXq6qqSp///Oe1evXqQY+zLEtz5szRihUrdPDBB2uvvfbSfffdJ2lPOIzFYjr44INVWVmpUaNG6YILLtDf/va3lO9x//3369RTT9XYsWNVUVGh/fffX1deeWXOjtrq1atlWZb+53/+Z9Dn/uM//kOWZWn9+vWSpLffflszZ85Uc3OzKioqNGbMGJ1xxhl65ZVXsv6MGTNmaPjw4dq8ebNOPvlkDRs2TGPHjlVHR4d2796dfFzittAlS5YoFotp//33V1lZmdauXZvxltE///nPOvvss9XY2Kiqqip99rOf1eWXX57ymDfeeEOzZs3SqFGjVFFRoYMPPlh33nln1jEDALxBIASAEpIIhOPHj9c3v/lNLVu2TN/97ndlWZYuueSSlMfOmDFDlmXpW9/6lm6++WZNnz5dlmVp8uTJKY+zGwgvv/xyRSIRfe9739Ptt9+upUuXatq0aVq8eHHyMc8995xqa2t18MEH69prr9VNN92kCRMmKBKJ6N577835/FpaWjR79mzddNNNuv7663X00UfLsiz913/916Axjxs3Tk1NTero6NDNN9+stWvXSpK++93vaq+99tL3vvc9/cd//IcuvfRSDR8+XEcddZQ++eST5PeYPHmypkyZoiVLluiWW27R2WefLcuy9KMf/SjrGHfs2KHq6mrNnj170OcmTpyoQw45JPnfxx57rGpra7VgwQLdcccduuaaazRx4kT99re/zfozZsyYoaqqKh144IH6zne+o5tuukmnn366LMvSwoULk49LhL6DDz5Y+++/vxYvXqxYLKbXXnstbSBct26dampq1NDQoPnz5+vWW2/VJZdcosMOOyz5mHfeeUctLS1qbW3VlVdeqVtuuUVnnHGGLMtSLBbLOm4AgHkEQgAoIYlAeN5556V8/Mwzz1RDQ0Pyv5955hlZlqXvfve7KY/70Y9+JMuy9Jvf/Cb5MbuB8IgjjtBpp52WdXwnnniiDjvssJQu5O7du3XsscfqwAMPzPn8duzYkfLfn3zyiQ499FD94z/+Y8rHLctSWVmZNmzYkPLx3/3ud7IsS7/4xS9SPv7II48M+vjAnyVJF154oYYNGzaoizrQtGnTNGrUqJRu4ttvv62ysjJdeeWVkqQPPvgg2b3LVyLM/+AHP0h+bPfu3TrttNNUUVGRvGU4Efpqamr0l7/8JeV7pAuEEyZM0IgRI/Taa6+lPLZ/1/H888/X2LFj9d5776U85pxzzlFtbW3a1w0A4B0CIQCUkEQgfOqpp1I+fv3118uyLG3btk2SdM0118iyLG3cuDHlcW+//bYsy9K8efOSH7MbCI8//nh95jOf0aZNm9KO7f3331ckEtGPf/xj/fWvf025Ojo6ZFmW3njjDdvP9W9/+5v++te/6vvf/77q6upSPmdZliZOnDjoay6++GLV1tbqL3/5y6AxVFdXDwrICR9++KH++te/asWKFbIsS88880zWsd1///2yLEs9PT3Jj/37v/+7LMvSCy+8IEnauXOnKioqdNpppw26XTWXRCBMfK+Ehx9+WJZlaeXKlZL+HvpmzZo16HsMDIR/+ctfZFmW5s6dm/Hn7t69W3V1dbrgggsGvX533XWXLMvS73//+7yeCwDAXQRCACghiUD4zjvvpHw8Uay/+uqrkvZ0usrKylJukUyoq6vTt771reR/2w2Ev/3tb1VXVyfLsnTooYfqRz/6kdatW5f8/JNPPinLsrJeTz/9dNbn9+CDD+pLX/qSKisrU74uEomkPC5dl1SSvvGNb2T9+WeccUbysc8995wmT56smpqaQY/LdUvnzp07VVtbq+9973vJj33lK1/RkUcemfK4WCymsrIyDRkyRF/96ld17bXX6u233876vaU9gbCsrEy7du1K+fjmzZtlWZZ+8pOfSPp76Et0JfsbGAj/+Mc/yrIs3X777Rl/7rvvvpvzd2jn1l8AgDkEQgAoIZl2GU0EwsRmJYlAODBQSPYDYUtLS0oglPZ0AX/2s5/pnHPOUV1dncrLy5MB4//+7/+Sa/Aef/zxtNeHH36Y8bn97//+ryKRiI4//njdeeedeuihh/T444/r29/+tiwr9f/dJTaVGejrX/+6Ro0alfHnJzp/H3zwgRoaGrTffvvppz/9qR588EE9/vjjuvbaa2VZlnp7ezOOM2HGjBlqbGzUrl279MYbbygSiSSDWn8vvfSSrrvuOp100kmqqKhQXV1dzmCcbyBMd1tqIYEw0UH+53/+54yv4bvvvpvrpQEAGEQgBIASYjcQZrpl9J133hl0y+jIkSMH3UYYj8dVXl4+KBD2t337do0fP17Nzc2S/t5dmj9/fkHPbe7cuRo6dOig9Xv5BMLZs2ervLw85zq3++67L20n8LbbbrMdCB966CFZlqVHHnlEsVhMlmXp5Zdfzvo1mzZt0rBhw3TuuedmfVy+t4zaCYR2bhnt6+vTiBEjNG3atKzjAwD4B4EQAEqI3UCY2FTmggsuSHncJZdcMmhTmS9+8YsaP358yuMS6+H6B8KBm4xISh5fkHDCCSeovr5eb7311qDHDtz0ZKB/+Zd/0bBhw1KOz3jllVc0bNgw24Hwf/7nfzKG0l27dumDDz6QJP3qV78adHREPB7XkUceaTsQfvLJJ6qvr9esWbN0zDHH6Oijj075/EcffTTobMRPP/1Uo0ePTunQppNtU5khQ4YkX8t8AqFkb1OZmTNnqqKiInl0Rn+5focAAPMIhABQQuwGQunvoWLKlCm6+eabk/898NiJxNl53/zmN3XLLbfooosu0n777afGxsaUQDhq1ChNmTJF1157rW6//XZdeOGFikQiKaFlw4YNGjlypBoaGnTZZZfptttu049//GOdeuqpOvzww7M+t1//+teyLEtf/epXdcstt6ijo0OjRo3S4YcfbjsQSntul7UsS9/4xjcUi8V00003ae7cudp7772TZxq+9957GjlypPbdd18tXbpU119/vcaPH68jjjjCdiCU9hxxUV1drUgkoqVLl6Z8bu3ataqvr9dFF12kG2+8UcuWLdNJJ50ky7L0y1/+Muv37X/sxPTp03XzzTcnj53of2ZgvoHwmWeeUXV1dfLYidtuu02XX365jjjiiORj3nnnHe27774aNmyY5s6dq1tvvVU/+clPdPbZZ2vkyJG2XhcAgDkEQgAoIfkEwl27dqmjo0P77befhgwZotbW1rQH03/66ae69NJL1djYqGHDhunrX/+6XnrppUGbylx11VU6+uijVVdXp6FDh+qggw7S1VdfPWjjms2bN2v69OkaM2aMhgwZoubmZp1++uk5Q5Ak3XnnnTrwwANVWVmpgw46SHfddVfyOfeXLRBKe279/MIXvqChQ4dqxIgROuyww3TJJZekdC6feOIJHXPMMRo6dKj23ntvXXLJJXr00UfzCoSPP/54ctObLVu2pHzuvffe05w5c3TQQQdp+PDhqq2t1Ze+9CV1dXXl/L7pDqYfPXq0Fi1apE8//TT5uHwDobRnM50zzzxTdXV1qqqq0uc+97mUsw2lPbf/zpkzR62trRoyZIjGjBmjE088Ubfddput1wUAYA6BEACAkEkEQgAAciEQAgAQMgRCAIBdBEIAAEKGQAgAsItACABAyBAIAQB2EQgBAAAAoEQRCAEAAACgRBEIAQAAAKBEEQgDbPfu3dq6dau2bNmirVu3atu2bVxcXFxcXFxcXFyBvxI1bv+zU+EOAmGAbdu2TZZlcXFxcXFxcXFxcYXy2rJli9cld+gRCANs9+7d2rJlS3KyeP1ODhcXFxcXFxcXF5cTV6LG3bp1q9cld+gRCANu27Y9XcJt27Z5PRQAAADAEdS45hAIA47JAgAAgLChxjWHQBhwTBYAAACEDTWuOQTCgGOyAAAAIGyocc0hEAYckwUAAABhQ41rDoEw4JgsAAAACBtqXHMIhAHHZAEAAEDYUOOaQyAMOCYLAAAAwoYa1xwCYcAxWQAAABA21LjmEAjT+O1vf6vTTz9dY8eOlWVZuu+++1I+v3v3bi1cuFBjxoxRVVWVTjzxRG3atCnlMe+//76+/e1va8SIEaqtrdV5552n7du3pzxm3bp1+spXvqLKykq1tLTo2muvzXusTBYAAACEDTWuOQTCNB566CH967/+q+699960gXDx4sWqra3V/fffr3Xr1umMM87Qfvvtp48//jj5mFNOOUVHHHGE/vjHP+p3v/udDjjgAE2bNi35+W3btmn06NE699xz9dxzz2nlypUaOnSobr311rzGymQBAABA2FDjmkMgzGFgINy9e7fGjBmjJUuWJD+2detWVVZWauXKlZKkjRs3yrIs/elPf0o+5uGHH1YkEtGbb74pSVq2bJlGjhypeDyefMyll16qz33uc3mNj8kCOKuvr0+9vb3q7OxUb2+v+vr6vB4SAAAlhxrXHAJhDgMD4ebNm2VZltauXZvyuAkTJujiiy+WJN15552qq6tL+fyuXbtUXl6ue++9V5L0ne98R5MmTUp5zG9+8xtZlqW//e1vGcezc+dObdu2LXlt2bKFyQI4pLu7Wy0tLbIsK3m1tLSou7vb66EBAFBSCITmEAhzGBgIn3jiC
