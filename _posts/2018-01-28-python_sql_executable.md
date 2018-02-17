---
title: Deep Learning with Database as Executable file
updated: 2018-01-28 22:35
layout: post
author: Wonik Jang
category: Deep Learning with Database as Executable file
tags:
- Executable
- SQL
- Classification
- Convolutional Neural Network
- Python
- Tensorflow
---

# **Deep Learning Model + SQL Database + Threading**

The circumstance that I encountered was to construct a program, which can run by itself and should keep watching multiple Database tables with 10 sec and 30 sec intervals. To make it possible, I used thread to check whether new data tables are updated. If new data is found, then I access to the data while remembering previously processed one. After transforming the new scanned data into 2D matrix and treat it as an image, I implemented trained CNN model to predict the grade of a product. Finally, I write the prediction result with corresponding information on an outcome table and saved image into path(written in Json file).

# **Structure of Codes**

# 1. Prerequisites
- Global Variables: Required information to access SQL
- Json file: Paths where to save image and where trained model parameter located

# 2. Classes
- Class A: SQL manager
- Class B: Deep Learning Model and Pre-Processing
- Function C: scheduler, which call function in Class A and Class B to check whether a new data came in and execute to save image and perform prediction.

# 3. Main function
- Class A initialization enables us to connect database.
- Class B initialization read Json file for necessary paths
- SetModel function in Class B ensures whether trained model parameters are loaded
- Catch for Function C


{% highlight ruby %}

import treading
import logging

# 1.1 Global Varaibles such as localhost, id, and password to access SQL database
sql_host = 'tcp:xxx,xxx,xx,xx'
sql_user = 'wonik'
sql_password = 'jang'
sql_db = 'pro'
sql_encode = 'utf-8'

# 1.2 Load Json file, which contains relevant paths  
abs_path = './'
json0 = 'path_inform_final.json'

for root, dirs, files in os.walk( abs_path ):
    for name in files:
        if name == json0:
            json_path0 = os.path.abspath(os.path.join(root, name))

json_path1 = json_path0.replace('\\','/')

# 2.1 Class A

# query sql command within appropriate functions
Class SqlManager(object):
    def __init__(self):
        # Set Connection)
        self.conn = pymssql.connect(server = sql_host[4:], user = sql_user,
                                    password = sql_password, database = sql_db)
    def __del__(self):
    def GetRecentProcData(self):
    def MonitorAdb(self):
    def MonitorBdb(self):
    def UpdateResultAdb(self):
    def UpdateResultBdb(self):

# 2.2 Class B
Class PreprocDL(object):
    def __init__(self, json_path1):
    def SetModel(self):
    def Normalize(self, data, w, h):
    def PosToIndex(self, pos):
    def ReadFile(self, identifier, data, additional):
    def PreProcess(self, identifier, data, additional):
    def ImageSaver(self, identifier, data, additional):
    def DoClassify(self, identifier, data, additional):

# 2.3 Function C
def scheduler1():
    # Load a function in Class A, which monitor Database
    ...
    # if NewData is not empyty, execute pre-processing and prediction with loaded model
    ...
    # run timer
    thread0 = threading.Timer(CHK_TIME, scheduler1)
    thread0.start()

# 3. Main funciton
CHK_TIME = 10
if __name__ == "__main__":

    # SQL Manager
    SqlManager = SqlManager()

    # Classifier
    MyClf = PreprocDL(json_path1)

    # Load trained model paramerters  
    init_res = MyClf.SetModel()

    # SetModel() returns init_res as an indicator of model loading by Catch

    if init_res == 0:

       try :
           scheduler()

       except Exception as err:
           logging.exception("Error!")
   else :
       logging.debug("\nDeep Learning Model Loading Fail...!\n" )
       # raise Exception
       raise RuntimeError


{% endhighlight %}

# **Build executable file(.exe) using pyinstaller**

Things to remember
1. If Tensorflow environment is CPU(GPU) based, trained model should be built upon CPU(GPU)
2. Type in required packages when building an exe file

Example of generating one python executable file in command line.

pyinstaller --hidden-import=scipy.integrate --hidden-import=scipy.integrate.quadpack --hidden-import=scipy._lib.messagestream --hidden-import=scipy.integrate._vode -F filename.py
