import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import sklearn
import pickle
import numpy as np

pipe=pickle.load(open('ppp.pkl','rb'))
df=pickle.load(open('ddd.pkl','rb'))

st.title('Laptop Price Predictor')

ram=st.selectbox("Ram",options=df["RAM"].unique())

def r(ram):
    if ram == 32:
        return 4
    elif ram == 64:
        return 3
    elif ram == 16:
        return 2
    elif ram == 8:
        return 1
    
laptop_ram=r(ram)

processor_type= st.selectbox("Name of the Processor",options=df["processor_type"].unique())

def pro_name(processor_type):
    
	if processor_type=="M1":
		return 14
	elif processor_type=="i9":
		return 13
	elif processor_type=="M2":
		return 12
	elif processor_type=="Ryzen 9":
		return 11
	elif processor_type=="i7":
		return 10
	elif processor_type=="Ryzen 5":
		return 9
	elif processor_type=="Ryzen 7":
		return 8
	elif processor_type=="i5":
		return 7
	elif processor_type=="i3":
		return 6
	elif processor_type=="Ryzen 3":
		return 5
	elif processor_type=="Athlon Dual":
		return 3
	elif processor_type=="Celeron Quad":
		return 2
	elif processor_type=="Pentium":
	   return 1
processor_name=pro_name(processor_type)

size_in_inches=st.selectbox("Size of the laptop in Inches",options=df["Inch"].unique())

diskdrive_type = st.selectbox("Disk Drive",options=df["disk_drive"].unique())
def disk_type(diskdrive_type):
    if diskdrive_type  == "SSD":
        return 2
    elif diskdrive_type == "HDD":
        return 1
    elif diskdrive_type  == "Both":
        return 3

type_of_diskdrive = disk_type(diskdrive_type)



size = st.selectbox("Disk size",options=df["Disc_size"].unique())
def disk_size(size):
    if size  == "2":
        return 11
    elif size == "1,512":
        return 10
    elif size  == "1,256":
        return 9
    elif size  == "256,256":
        return 8
    elif size  == "1 SSD":
        return 7
    elif size  == "512":
        return 6
    elif size  == "1 HDD":
        return 5
    elif size  == "256":
        return 4
    elif size  == "128":
        return 3
    elif size  == "64":
        return 2
    elif size  == "32":
        return 1

type_disksize = disk_size(size)

range_price=st.selectbox("Range of price",options=df["Price_range"].unique())

def price(range_price):
    if range_price  == "low":
        return 1
    elif range_price == "moderate":
        return 2
    elif range_price  == "high":
        return 3
range_of_price=price(range_price)

waranty=st.selectbox("warranty of laptop",options=df["Warranty"].unique())
def w(waranty):
    if waranty  == "1":
        return 1
    elif waranty == "2":
        return 2
    elif waranty  == "3":
        return 3
    

features=[laptop_ram,processor_name,size_in_inches,type_of_diskdrive,type_disksize,range_of_price,waranty]

final_features=np.array(features).reshape(1,-1)

if st.button('Predict'):
     prediction=pipe.predict(final_features)	
     st.balloons()
     st.success(f'Your predicted price of the laptop is {prediction[0]}')