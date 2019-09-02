import time
import os
import numpy as np
from scipy import ndimage
from ctypes import *
import sys
import h5py
from threading import Thread
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.optimize import leastsq, curve_fit, basinhopping


class Datafitting:
    def __init__(self):
        #指定拟合文件位置
        path0 = os.getcwd()   #获取当前路径
        self.path_to_file = str(path0 + '/CCD测温度/21h-14m-59s_data.txt')
        #print(path)

        #初始化数据
        self.a_x = None
        self.a_y = None
        self.b_x = None
        self.b_y = None
        self.a_fitting = None
        self.a_fitting_x = None
        self.b_fitting = None
        self.b_fitting_x = None

        self.popt_a = None
        self.popt_b = None

    def curve_function_gauss(self, x, y0 , xc, w, A):
        '''高斯函数拟合公式和参数设置'''
        return y0 + (A/(w*np.sqrt(np.pi/2)))*np.exp(-2*((x-xc)/w)**2)

    def calculate_p0(self, x, y):
        '''计算初值'''
        y0 = 0
        xc = x[np.argmax(y)]
        w = 1
        A = np.max(y)
        return y0 , xc, w, A

    def fitting(self):
        '''拟合进程，返回拟合参数'''
        p0 = self.calculate_p0(self.a_x,self.a_y)
        popt_a , pvoc = curve_fit(self.curve_function_gauss, self.a_x, self.a_y, p0)
        #获取拟合数据a
        self.a_fitting = self.get_fit_data(popt_a, self.a_x)
        popt_a = self.covert_to_float(popt_a)

        p0 = self.calculate_p0(self.b_x,self.b_y)
        popt_b , pvoc = curve_fit(self.curve_function_gauss, self.b_x, self.b_y, p0)
        #获取拟合数据b
        self.b_fitting = self.get_fit_data(popt_b, self.b_x)
        popt_b = self.covert_to_float(popt_b)
        
        return popt_a , popt_b

    def get_fit_data(self, p, x):
        '''获取拟合后参数'''
        xmin = np.min(x)
        xmax = np.max(x)
        x = np.linspace(xmin, xmax, 500)
        y0 , xc, w, A = p
        return self.curve_function_gauss(x, y0, xc, w, A)

    def get_data(self,path):
        '''获取待拟合数据'''
        with open(path, 'r') as f:
            raw_data = f.readlines()
            self.a_x = np.full(0,0)
            self.a_y = np.full(0,0)
            self.b_x = np.full(0,0)
            self.b_y = np.full(0,0)
            for i in range(len(raw_data) - 2):
                #print(raw_data[i + 2].split('\t'))
                if raw_data[i + 2].split('\t')[2] != '\n':
                    #
                    #isinstance(raw_data[i + 2].split('\t')[2], int)
                    self.a_x = np.append(self.a_x,float(raw_data[i + 2].split('\t')[0]))
                    self.a_y = np.append(self.a_y,float(raw_data[i + 2].split('\t')[1]))
                    self.b_x = np.append(self.b_x,float(raw_data[i + 2].split('\t')[2]))
                    self.b_y = np.append(self.b_y,float(raw_data[i + 2].split('\t')[3].split('\n')[0]))
                else:
                    self.a_x = np.append(self.a_x,float(raw_data[i + 2].split('\t')[0]))
                    self.a_y = np.append(self.a_y,float(raw_data[i + 2].split('\t')[1]))
            '''
            #
            for i in (self.a_x , self.a_y , self.b_x , self.b_y):
                for j in range(len(i)):
                    i[j] = float(i[j])
            '''

            #print(self.a_x,self.a_y,self.b_x,self.b_y)
            #print(f.readlines())
            #return self.a_x,self.a_y,self.b_x,self.b_y

    def covert_to_float(self,p):
        '''转换数据类型'''
        for i in range(len(p)):
            p[i] = float(p[i])
            #print(p[i])
        return p
    
    def create_file(self, file_name='data'):
        #time = time.strftime("%Y-%m-%d", time.localtime())
        year = time.strftime("%Y", time.localtime())
        month = time.strftime("%m", time.localtime())
        date = time.strftime("%d", time.localtime())
        name_time = time.strftime('%Hh-%Mm-%Ss', time.localtime())
        file_name = name_time + '_'+ file_name
        path = 'E:/ChenZ/Datafitting/'+ year +'/'+ month +'/'+ date +'/'
        if not os.path.exists(path):
            os.makedirs(path)
        file = open(path + file_name +'.txt', 'w+')
        return file

    def write_file(self, file):
        #f = self.create_file('fitting')
        file.write('axial\t'+'y0='+str(self.popt_a[0])+'\t'+'xc='+str(self.popt_a[1])+'\t'+\
            'w='+str(self.popt_a[2])+'\t'+'A='+str(self.popt_a[3])+'\n')
        file.write('radial\t'+'y0='+str(self.popt_b[0])+'\t'+'xc='+str(self.popt_b[1])+'\t'+\
            'w='+str(self.popt_b[2])+'\t'+'A='+str(self.popt_b[3])+'\n')
        file.flush()

        return file

    def run(self):
        f = self.create_file('fitting')
        
        '''
        #for test
        self.path_to_file
        self.get_data(self.path_to_file)
        self.popt_a, self.popt_b = self.fitting()
        f.write('File: '+ self.path_to_file.split('/')[-1]+'\n')
        self.write_file(f)
        '''
        
        #获取目标文件夹的路径
        meragefiledir = os.getcwd()+'/CCD测温度'
        #获取当前文件夹中的文件名称列表
        filenames=os.listdir(meragefiledir)
        #打开当前目录下的result.txt文件，如果没有则创建
        #文件也可以是其他类型的格式，如result.js
        file=open('result.txt','w')
        #先遍历文件名
        for filename in filenames:
            filepath = meragefiledir+'\\'+filename
            #对单个文件进行数据处理拟合
            self.get_data(filepath)
            self.popt_a, self.popt_b = self.fitting()
            f.write('Time:'+filename.split('_data')[0]+'\n')
            f = self.write_file(f)
            #遍历单个文件，读取行数
            for line in open(filepath):
                file.writelines(line)
            file.write('\n')
        #关闭文件
        file.close()
        

if __name__ == '__main__':
    fit = Datafitting()
    fit.run()