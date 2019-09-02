import time
import os

class DataAnaly:
    def __init__(self):
        #指定文件位置
        self.folder = os.getcwd()  #获取当前路径
        self.subfolder = self.folder + '/CCD测温度-0801-1' 
        #print(self.folder)
        #self.path_to_file = str(self.folder + '/CCD测温度-0618/0.80/17h-15m-59s_data.txt')
        self.volt = 0.0
        self.w_a = 0.0
        self.w_r = 0.0

    def covert_to_float(self,p):
        '''转换数据类型'''
        #p = list(p)
        for i in range(len(p)):
            p[i] = float(p[i])
            #print(p[i])
        return p

    def get_data(self, path):
        """获取文件中所需数据"""
        #print(path)
        with open(path, 'r') as f:
            data = f.readlines()
            line1 = str(data[0])
            line2 = str(data[1])
            volt = str(path).split('/')[-1].split("\\")[0]
            w_a = line1.split('\t')[3].split('=')[-1]
            w_r = line2.split('\t')[3].split('=')[-1]
            #print(volt, w_a, w_r)
        return volt, w_a, w_r


    def create_file(self, file_name):
        year = time.strftime("%Y", time.localtime())
        month = time.strftime("%m", time.localtime())
        date = time.strftime("%d", time.localtime())
        name_time = time.strftime('%Hh-%Mm-%Ss', time.localtime())
        file_name = name_time + '_'+ file_name
        path = self.folder + '/DataAnaly/'+ year +'_'+ month +'_'+ date +'_'
        '''
        if not os.path.exists(path):
            os.makedirs(path)
        '''
        file = open(path + file_name +'.txt', 'w+')
        file.write('Secular Freq'+'\t'+'w-axial'+'\t'+'w-radial'+'\n')
        return file

    def write_file(self, file):
        file.write(str(self.volt).split('=')[-1]+'\t'+str(self.w_a).split('=')[-1]
            +'\t'+str(self.w_r).split('=')[-1]+'\n')
        file.flush()
        return file

    def run(self):
        f = self.create_file('DataAnaly')
        
        list_dirs = os.walk(self.subfolder)
        for root, dirs, files in list_dirs:
            for d in dirs:
                filenames = os.listdir(self.subfolder + '/' + d)
                #print(filenames)
                for filename in filenames:
                    filepath = self.subfolder + '/' + d +'\\'+filename
                    volt, w_a, w_r = self.get_data(filepath)
                    #self。volt = float(volt)
                    self.volt = volt
                    self.w_a += float(w_a)
                    self.w_r += float(w_r)
                #self.volt = self.covert_to_float(volt)
                #print(len(filenames))
                self.w_a = self.w_a/len(filenames)
                self.w_r = self.w_r/len(filenames)
                #print(volt,w_a,w_r)
                f = self.write_file(f)
                    
        '''
        filepath = self.folder
        self.volt, self.w_a, self.w_r = self.get_data(filepath)
        self.write_file(f)
        '''

        #关闭文件
        f.close()
        

if __name__ == '__main__':
    analy = DataAnaly()
    analy.run()