import time
import os
import numpy as np
from scipy import ndimage
from andorCCD import AndoriXon
from ctypes import *
import sys
import h5py
from threading import Thread
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.optimize import leastsq, curve_fit, basinhopping


class CCDViewer:
    def __init__(self):
        self.Gain = 50
        self.exposure_time = 0.5
        self.file_center = self.create_file()
        self.file_full_view = self.create_file('full_veiw')
        self.prepare()
        self.init_pic()
        #self.init_pic_4()

        #用来存储x,y方向的原始及拟合数据
        self.x_x = None
        self.y_x = None
        self.x_raw = None
        self.y_raw = None
        self.y_fitting = None
        self.x_fitting = None
        self.x_fitting_x = None
        self.y_fitting_x = None

        self.popt_x = None
        self.popt_y = None

        self.line_x_0 = None
        self.line_x_1 = None
        self.line_y_0 = None
        self.line_y_1 = None
        
    def init_pic(self):
        plt.subplots_adjust(bottom=0.2)
        file = h5py.File('data2.h5', 'r')
        data = file['data'].value
        self.ax1 = plt.subplot(221)
        self.img1 = plt.imshow(data)

        data_filter = ndimage.gaussian_filter(data, sigma=1)
        binary_data = data_filter > data_filter.mean() + 6*data_filter.std()
        close_img = ndimage.binary_closing(ndimage.binary_opening(binary_data).astype(np.int))

        self.ax2 = plt.subplot(222)
        self.img2 = plt.imshow(close_img)
    
    def init_pic_4(self):
        self.ax3 = plt.subplot(223)
        self.line_x_0 = plt.plot(np.full(100,0))[0]
        self.line_x_1 = plt.plot(np.full(100,0))[0]
        self.ax4 = plt.subplot(224)
        self.line_y_0 = plt.plot(np.full(100,0))[0]
        self.line_y_1 = plt.plot(np.full(100,0))[0]
        self.line_x_0.set_linestyle('None')
        self.line_x_0.set_marker('*')
        self.line_y_0.set_linestyle('None')
        self.line_y_0.set_marker('*')


        # self.line_x_0.set_ydata(np.full(100,1))
        # self.line_x_1.set_ydata(np.full(100,2))
        # self.line_y_0.set_ydata(np.full(100,3))
        # self.line_y_1.set_ydata(np.full(100,4))
        # self.ax3.set_ylim(0,4)
        # self.ax4.set_ylim(0,5)
        #plt.draw()
        
        
    def prepare(self):
        self.ccd = AndoriXon(Gain=self.Gain, exposuretime=self.exposure_time)
    
        # 不要采集太大的区域 目前 socket 的传输速度被限制在 25 kB/s , artiq 3.0 版本可能能解决该问题     
        self.ccd.SetImage(4,4,301,700,301,700)
        self.ccd.GetTemperature()
        # while self.ccd.temperature > -65:
        #     self.ccd.GetTemperature() 
        #     time.sleep(self.exposure_time)
        #     print(self.ccd.temperature)
            
    def create_file(self,file_name='ccd'):
        #year = time.strftime("%Y-%m-%d", time.localtime())
        year = time.strftime("%Y", time.localtime())
        month = time.strftime("%m", time.localtime())
        data = time.strftime("%d", time.localtime())
        name_time = time.strftime('%Hh-%Mm-%Ss', time.localtime())
        file_name = name_time + '_'+ file_name
        path = 'D:/data/ccd/'+ year +'/'+ month +'/'+ data +'/'
        if not os.path.exists(path):
            os.makedirs(path)
        file = open(path + file_name +'.txt', 'w+')
        return file

    def check_ion(self, image):
        """ 采用图像学手段提取离子特征，计算离子位置
        
        方法：
            高斯平滑、二值化、去噪点、然后提取闭合区域信息
        返回值：
            离子数目和离子中心位置，对于多离子，是多个离子的中心位置
        """
        im = ndimage.gaussian_filter(image, sigma=1) # 平滑
        binary_image = im > im.mean() + 4*im.std() # 二值化
        close_img = ndimage.binary_closing(ndimage.binary_opening(binary_image).astype(np.int)) # 去除过小的闭合区域，并圆滑边缘（降低粘连）
        label_im, num_labels = ndimage.label(close_img) # 标记闭合区域
        if num_labels == 0:
            positions = [50, 50]
        else:
            positions = np.average(ndimage.center_of_mass(im, label_im, [i+1 for i in range(num_labels)]), 0) # 多离子时取离子平均位置
            
        ion_number = num_labels
        #center_positions = np.ceil(positions)
        
        return ion_number, positions, close_img
        
    def check_ion_fullveiw(self, image):
        image1 = np.full(400*400,0).reshape(400,400)
        for i in range(400):
            for j in range(400):
                image1[i,j] = image[i+300,j+300]
        image = image1
        im = ndimage.gaussian_filter(image, sigma=1) # 平滑
        binary_image = im > im.mean() + 8*im.std() # 二值化
        close_img = ndimage.binary_closing(ndimage.binary_opening(binary_image).astype(np.int)) # 去除过小的闭合区域，并圆滑边缘（降低粘连）
        label_im, num_labels = ndimage.label(close_img) # 标记闭合区域
        if num_labels == 0:
            positions = [200, 200]
        else:
            positions = np.average(ndimage.center_of_mass(im, label_im, [i+1 for i in range(num_labels)]), 0) # 多离子时取离子平均位置
        
        ion_number = num_labels
        
        positions[0] += 300
        positions[1] += 300
        
        #print(image.shape,close_img.shape)

        return ion_number, positions, close_img
        
    def get_intensity(self,image,close_img):
        intensity = 0
        num = 0
        m,n = np.shape(image)
        for i in range(m):
            for j in range(n):
                if close_img[i,j]:
                    intensity += image[i,j]
                    num += 1
                    #print(image[i,j])
        intensity -= num*image.mean()
        #print(image.mean())
        return intensity,intensity/num,num

    def get_data(self,image,close_img,position):
        #print('position: ',position)
        x0 = int(position[0])
        y0 = int(position[1])
        hx = 0
        hy = 0
        for i in range(400):
            if close_img[x0-300,i] > 0:
                hy += 1
            if close_img[i,y0-300] >0:
                hx += 1
        intensity = 0
        num = 0

        for i in range(int(x0-5*hx),int(x0-2*hx)):
            for j in range(int(y0-5*hy),int(y0-2*hy)):
                intensity += image[i,j]
                num += 1
        if num > 0:
            avg = intensity/num
        else:
            avg = 0

        x_array = np.full(0,0)
        y_array = np.full(0,0)
        for i in range(int(x0-2*hx),int(x0+2*hx)):
            x_array = np.append(x_array,image[i,y0]-avg)
        for i in range(int(y0-2*hy),int(y0+2*hy)):
            y_array = np.append(y_array,image[x0,i]-avg)

        self.x_x = np.arange(4*hx)
        self.y_x = np.arange(4*hy)
        self.x_raw = x_array
        self.y_raw = y_array

        # print(self.x_x)
        # print(self.x_raw)
        # print(self.y_x)
        # print(self.y_raw)

        return x0,y0,avg

    def fitting(self):
        p0 = self.calculate_p0(self.x_x,self.x_raw)
        popt_x , pvoc = curve_fit(self.curve_function_gauss, self.x_x, self.x_raw, p0)
        self.x_fitting = self.get_fit_data(popt_x, self.x_x)

        p0 = self.calculate_p0(self.y_x,self.y_raw)
        popt_y , pvoc = curve_fit(self.curve_function_gauss, self.y_x, self.y_raw, p0)
        self.y_fitting = self.get_fit_data(popt_y, self.y_x)

        return popt_x,popt_y
    
    def function_gauss(self, x, p):
        y0 , xc, w, A = p
        x = np.array(x)
        return y0 + (A/(w*np.sqrt(np.pi/2)))*np.exp(-2*((x-xc)/w)**2)

    def curve_function_gauss(self, x, y0 , xc, w, A):
        return y0 + (A/(w*np.sqrt(np.pi/2)))*np.exp(-2*((x-xc)/w)**2)

    def get_fit_data(self, p, x):
        xmin = np.min(x)
        xmax = np.max(x)
        x = np.linspace(xmin, xmax, 500)
        return self.function_gauss(x, p)

    def calculate_p0(self,x,y):
        xc = x[np.argmax(y)]
        A = np.max(y)
        return 0,xc,1,A

    def updata_fitting(self,image,close_img,positions):
        x0,y0,avg = self.get_data(image,close_img,positions)
        popt_x, popt_y = self.fitting()
        self.popt_x = popt_x
        self.popt_y = popt_y
        self.ax3.set_title('FWHM(axial): '+str(abs(1.1774*popt_x[2])))
        self.ax4.set_title('FWHM(radial): '+str(abs(1.1774*popt_y[2])))
        # print('size: x_x: ', len(self.x_x))
        # print('size: x_raw: ', len(self.x_raw))
        # print('size: x_fitting: ', len(self.x_fitting))
        # print('size: y_x: ', len(self.y_x))
        # print('size: y_raw: ', len(self.y_raw))
        # print('size: y_fitting: ', len(self.y_fitting))

        self.line_x_0.set_xdata(self.x_x)
        self.line_x_0.set_ydata(self.x_raw)
        self.line_x_1.set_xdata(np.linspace(np.min(self.x_x),np.max(self.x_x), 500))
        self.line_x_1.set_ydata(self.x_fitting)

        self.line_y_0.set_xdata(self.y_x)
        self.line_y_0.set_ydata(self.y_raw)
        self.line_y_1.set_xdata(np.linspace(np.min(self.y_x),np.max(self.y_x), 500))
        self.line_y_1.set_ydata(self.y_fitting)
        
        self.updata_lim()

    def updata_lim(self):
        y_data = self.line_x_0.get_ydata()
        y_step = np.nanmax(y_data) - np.nanmin(y_data)
        y_min = np.nanmin(y_data)-0.1*y_step
        y_max = np.nanmax(y_data)+0.1*y_step
        self.ax3.set_ylim(y_min,y_max)
        x_data = self.line_x_0.get_xdata()
        x_step = np.nanmax(x_data) - np.nanmin(x_data)
        x_min = np.nanmin(x_data)-0.1*x_step
        x_max = np.nanmax(x_data)+0.1*x_step
        self.ax3.set_xlim(x_min,x_max)

        y_data = self.line_y_0.get_ydata()
        y_step = np.nanmax(y_data) - np.nanmin(y_data)
        y_min = np.nanmin(y_data)-0.1*y_step
        y_max = np.nanmax(y_data)+0.1*y_step
        self.ax4.set_ylim(y_min,y_max)
        x_data = self.line_y_0.get_xdata()
        x_step = np.nanmax(x_data) - np.nanmin(x_data)
        x_min = np.nanmin(x_data)-0.1*x_step
        x_max = np.nanmax(x_data)+0.1*x_step
        self.ax4.set_xlim(x_min,x_max)

    def write_file(self,event):
        f = self.create_file('data')
        f.write('左右\t'+'y0='+str(self.popt_x[0])+'\t'+'xc='+str(self.popt_x[1])+'\t'+'w='+str(self.popt_x[2])+'\t'+'A='+str(self.popt_x[3])+'\t\n')
        f.write('上下\t'+'y0='+str(self.popt_y[0])+'\t'+'xc='+str(self.popt_y[1])+'\t'+'w='+str(self.popt_y[2])+'\t'+'A='+str(self.popt_y[3])+'\t\n')
        n_x = len(self.x_x)
        n_y = len(self.y_x)
        n = max(n_x,n_y)
        for i in range(n):
            if i < n_x:
                f.write(str(self.x_x[i])+'\t'+str(self.x_raw[i])+'\t')
            if i < n_y:
                f.write(str(self.y_x[i])+'\t'+str(self.y_raw[i])+'\t')
            f.write('\n')
        f.flush()

    def run(self):
        first_coming = True
        while self.flag:  
            if first_coming:
                self.ccd.SetImage(4,4,301,700,301,700) 
                imdata = self.ccd.GetData()
                if imdata.shape[0] != 100:
                    return
                num, positions, close_img = self.check_ion(imdata)
                del self.img1
                del self.img2
                plt.subplot(121)
                self.img1 = plt.imshow(imdata)
                plt.subplot(122)
                self.img2 = plt.imshow(close_img)
                first_coming = False
            else:
                imdata = self.ccd.GetData()
                if imdata.shape[0] != 100:
                    return
                num, positions, close_img = self.check_ion(imdata)
                self.img1.set_data(imdata)
                self.img2.set_data(close_img)
                

            if num == 1:
                yi = positions[0]
                xi = positions[1]
                intensity,avg,area = self.get_intensity(imdata,close_img)
                self.ax1.set_title('num: '+str(num)+"; intensity: "+str(intensity)[0:7]+"; avg: "+str(avg)[0:5])
                self.ax2.set_title("x: "+str(xi)[0:5]+',y: '+str(yi)[0:5]+',area: '+str(area))

                self.file_center.write( time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime()) )
                self.file_center.write('\t')
                self.file_center.write(str(time.time()))
                self.file_center.write('\t')
                self.file_center.write(str(yi)+'\t'+str(xi)+'\t'+str(intensity)+'\t'+str(avg)+'\n')
                self.file_center.flush()
            else:
                self.ax1.set_title('num: '+str(num))

            #print('num: ', num)
            plt.draw()
    
    def Start(self, event):
        self.flag =True
        self.flag_full_view = False
        if hasattr(self,'f_full'):
            if self.t_full.isAlive():
                time.sleep(0.6)
        #创建并启动新线程
        self.t =Thread(target=self.run)
        self.t.start()
    
    def Stop(self, event):
        self.flag =False
        
    def run_full_view(self):
        first_coming = True
        while self.flag_full_view:
            if first_coming:
                self.ccd.SetImage(1,1,1,1000,1,1000)
                imdata = self.ccd.GetData()
                if imdata.shape[0] != 1000:
                    return
                num, positions, close_img = self.check_ion_fullveiw(imdata)
                del self.img1
                del self.img2
                plt.subplot(221)
                self.img1 = plt.imshow(imdata)
                plt.subplot(222)
                self.img2 = plt.imshow(close_img)
                if self.line_x_0 is not None:
                    del self.line_x_0
                    del self.line_x_1
                    del self.line_y_0
                    del self.line_y_1
                self.init_pic_4()
                first_coming = False
            else:
                imdata = self.ccd.GetData()
                if imdata.shape[0] != 1000:
                    return
                num, positions, close_img = self.check_ion_fullveiw(imdata)
                self.img1.set_data(imdata)
                self.img2.set_data(close_img)
            
            if num >= 1:
                yi = positions[0]
                xi = positions[1]
                #intensity,avg,area = self.get_intensity(imdata,close_img)
                self.ax1.set_title('num: '+str(num))
                self.ax2.set_title("x: "+str(xi)[0:5]+',y: '+str(yi)[0:5])
                
                self.file_full_view.write( time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime()) )
                self.file_full_view.write('\t')
                self.file_full_view.write(str(time.time()))
                self.file_full_view.write('\t')
                self.file_full_view.write(str(xi)+'\t'+str(yi)+'\n')
            else:
                self.ax1.set_title('num: '+str(num))
            print('num: ',num)
            if num == 1:
                try:
                    self.updata_fitting(imdata, close_img, positions)
                    
                except:
                    print('error')
            
            plt.draw()
            #return
            
    def start_full_view(self, event):
        self.flag_full_view =True
        self.flag = False
        if hasattr(self,'t'):
            if self.t.isAlive():
                time.sleep(0.6)
        #创建并启动新线程
        self.t_full =Thread(target=self.run_full_view)
        self.t_full.start()
            
    def stop_full_view(self, event):
        self.flag_full_view = False
                   
if __name__ == '__main__':
    callback =CCDViewer()
    #创建按钮并设置单击事件处理函数
    axprev = plt.axes([0.81,0.05,0.1,0.075])
    bprev =Button(axprev,'Stop')
    bprev.on_clicked(callback.Stop)
    
    axnext = plt.axes([0.7,0.05,0.1,0.075])
    bnext =Button(axnext,'Start')
    bnext.on_clicked(callback.Start)
    
    axprev1 = plt.axes([0.1,0.05,0.1,0.075])
    bprev1 =Button(axprev1,'FullView_Start')
    bprev1.on_clicked(callback.start_full_view)
    
    axnext1 = plt.axes([0.21,0.05,0.1,0.075])
    bnext1 =Button(axnext1,'FullView_Stop')
    bnext1.on_clicked(callback.stop_full_view)

    axnext2 = plt.axes([0.31,0.05,0.1,0.075])
    bnext2 =Button(axnext2,'Save_data')
    bnext2.on_clicked(callback.write_file)

    plt.show()
                        
                        