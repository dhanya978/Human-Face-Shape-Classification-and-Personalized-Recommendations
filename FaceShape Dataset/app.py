import shutil
import os
import sys
from flask import Flask, render_template, request
import cv2  # working with, mainly resizing, images
import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import \
    tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt



# global b
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
   
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"
       
        shutil.copy("testing_set\\"+fileName, dst)
        image = cv2.imread("testing_set\\"+fileName)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('static/gray.jpg', gray_image)
        image=np.zeros((300,300),dtype="uint8")
        retval2,threshold2 = cv2.threshold(gray_image,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.rectangle(threshold2,(10,10),(40,40),(0, 0, 255), 1)
        cv2.imwrite('static/threshold.jpg', threshold2)
        
        verify_dir = 'static/images'
        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'faceshape-{}-{}.model'.format(LR, '2conv-basic')
    ##    MODEL_NAME='keras_model.h5'
        def process_verify_data():
            verifying_data = []
            for img in os.listdir(verify_dir):
                path = os.path.join(verify_dir, img)
                img_num = img.split('.')[0]
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                verifying_data.append([np.array(img), img_num])
                np.save('verify_data.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()
        #verify_data = np.load('verify_data.npy')

        
        tf.compat.v1.reset_default_graph()
        #tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 5, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')


        fig = plt.figure()
        accuracy=""
        str_label=" "
        suggestions=" "
        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]

            y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            # model_out = model.predict([data])[0]
            model_out = model.predict([data])[0]
            print(model_out)
            print('model {}'.format(np.argmax(model_out)))

            
            if np.argmax(model_out) == 0:
                str_label = 'HEART SHAPE'
                print("The predicted image of the face has HEART SHAPE with a accuracy of {} %".format(model_out[0]*100))                
                accuracy = "The predicted image of the face has HEART SHAPE  with a accuracy of {} %".format(model_out[0]*100)
                suggestions=["HAIRSTYLES:","Length :Long.","Styles :Shoulder length bob, deep side part with loose waves, layers upto collarbone, sleek lob, long blonde curled ends, short pixie","Bangs:curtain bangs, wispy bangs "," EYE GLASSES : Cat eye, Rectangle, Wayfare, Oval, browline","CONTOUR AND HIGHLIGHT:contour on the center of your jawline and on the temples,cheekbones to take some width of upperface , Hightlight the tip of your nose,forehead and cupid's bow ","EYEBROW SHAPE:high arches(short heart),low arches(longer heart)"]

                A=float(model_out[0])
                B=float(model_out[1])
                C=float(model_out[2])
                D=float(model_out[3])
                E=float(model_out[4])

                dic={'heart':A,'oblong':B,'oval':C,'round':D,'sqaure':E}
                algm = list(dic.keys()) 
                accu = list(dic.values()) 
                fig = plt.figure(figsize = (5, 5))  
                plt.bar(algm, accu, color ='maroon', width = 0.3)  
                plt.xlabel("Comparision") 
                plt.ylabel("Accuracy Level") 
                plt.title("Accuracy Comparision between face shapes....")
                plt.savefig('static/matrix.png')
                
            elif np.argmax(model_out) == 1:
                str_label = 'oblong'
                print("The predicted image of the face has oblong shape with a accuracy of {} %".format(model_out[1]*100))
                accuracy = "The predicted image of the face has oblong shape with a accuracy of {} %".format(model_out[1]*100)
                suggestions=["HAIRSTYLES:","Length :Short or Medium."," Styles :Loss curles, flat iron waves, long layers, wavy shoulder length bob, bob with volume, tousled ponytail, deep side part","Bangs:curtain bangs, side swept bangs "," EYE GLASSES : Wayfare, Oval, browline, aviators, round, geometric "," CONTOUR AND HIGHLIGHT:contour to the top and bottom: the hairline and just below the chin , Hightlight under the eyes in a triangle shape sweeping it out towards your ears "," EYEBROW SHAPE:flat eyebrow"]
                A=float(model_out[0])
                B=float(model_out[1])
                C=float(model_out[2])
                D=float(model_out[3])
                E=float(model_out[4])

                dic={'heart':A,'oblong':B,'oval':C,'round':D,'sqaure':E}
                algm = list(dic.keys()) 
                accu = list(dic.values()) 
                fig = plt.figure(figsize = (5, 5))  
                plt.bar(algm, accu, color ='maroon', width = 0.3)  
                plt.xlabel("Comparision") 
                plt.ylabel("Accuracy Level") 
                plt.title("Accuracy Comparision between face shapes....")
                plt.savefig('static/matrix.png')
                
            elif np.argmax(model_out) == 2:
                str_label = 'oval'
                print("The predicted image of the face has oval shape with a accuracy of {} %".format(model_out[2]*100))
                accuracy = "The predicted image of the face has oval shape with a accuracy of {} %".format(model_out[2]*100)
                suggestions=["HAIRSTYLES:","Length :Medium or Short.","Styles :Blunt bob above the shoulder, Long layers, Shoulder length cut, side parted lob, sleek hairstyles, pixie cut, armpit length haircut with layered ends","Bangs:side swept bangs, baby bangs","EYE GLASSES : Round, Cat eye, Rectangle, Wayfare, Square, Oval, Aviators, Geometric, browline","CONTOUR AND HIGHLIGHT:contour under your cheekbones and on the outer edges of your forehead , Highlight on your highest points like cheekbones,brobone,bridge of your nose and chin "," EYEBROW SHAPE:soft angles and shallow arches"]
                A=float(model_out[0])
                B=float(model_out[1])
                C=float(model_out[2])
                D=float(model_out[3])
                E=float(model_out[4])

                dic={'heart':A,'oblong':B,'oval':C,'round':D,'sqaure':E}
                algm = list(dic.keys()) 
                accu = list(dic.values()) 
                fig = plt.figure(figsize = (5, 5))  
                plt.bar(algm, accu, color ='maroon', width = 0.3)  
                plt.xlabel("Comparision") 
                plt.ylabel("Accuracy Level") 
                plt.title("Accuracy Comparision between face shapes....")
                plt.savefig('static/matrix.png')
                
            elif np.argmax(model_out) == 3:
                str_label = 'round'
                print("The predicted image of the face has round shape with a accuracy of {} %".format(model_out[3]*100))
                accuracy = "The predicted image of the face has round shape with a accuracy of {} %".format(model_out[3]*100)
                suggestions=["HAIRSTYLES:","Length :Medium", "Styles :textured lob, slick back high ponytail, side parted pixie cut, long layers, symmetrical lengths, long bob, deep side part"," Bangs:blunt bangs, thick bangs ","  EYE GLASSES : Rectangle, Square,  Aviators, Geometric "," CONTOUR AND HIGHLIGTH:apply bronzer to your hairline and jawline , Highlight on your forehead,chin and cheekbones", "EYEBROW SHAPE:a lifted,angled arch"]
                A=float(model_out[0])
                B=float(model_out[1])
                C=float(model_out[2])
                D=float(model_out[3])
                E=float(model_out[4])

                dic={'heart':A,'oblong':B,'oval':C,'round':D,'sqaure':E}
                algm = list(dic.keys()) 
                accu = list(dic.values()) 
                fig = plt.figure(figsize = (5, 5))  
                plt.bar(algm, accu, color ='maroon', width = 0.3)  
                plt.xlabel("Comparision") 
                plt.ylabel("Accuracy Level") 
                plt.title("Accuracy Comparision between face shapes....")
                plt.savefig('static/matrix.png')
                
            elif np.argmax(model_out) == 4:
                str_label = 'sqaure'
                print("The predicted image of the face has sqaure shape with a accuracy of {} %".format(model_out[4]*100))
                accuracy = "The predicted image of the face has sqaure shape with a accuracy of {} %".format(model_out[4]*100)
                suggestions=["HAIRSTYLES:","Length :Medium or Long.","Styles :straight hair with long layers, soft waves, shoulder length bob, tousled lob, long and voluminous waves, blunt chin bob ","Bangs:soft,wispy side swept bangs, long layered bangs "," EYE GLASSES : Round, Cat eye,  Wayfare, Oval, Aviators,  browline", "CONTOUR AND HIGHLIGHT:add length to your face by applying bronzer on the outer corners of your forehead and jawline , Highlight your cheekbones,center of forehead,nose tip and chin "," EYEBROW SHAPE:a strong bro with a defined arch"]
                A=float(model_out[0])
                B=float(model_out[1])
                C=float(model_out[2])
                D=float(model_out[3])
                E=float(model_out[4])

                dic={'heart':A,'oblong':B,'oval':C,'round':D,'sqaure':E}
                algm = list(dic.keys()) 
                accu = list(dic.values()) 
                fig = plt.figure(figsize = (5, 5))  
                plt.bar(algm, accu, color ='maroon', width = 0.3)  
                plt.xlabel("Comparision") 
                plt.ylabel("Accuracy Level") 
                plt.title("Accuracy Comparision between face shapes....")
                plt.savefig('static/matrix.png')

            
                
            return render_template('home.html', status=str_label,accuracy=accuracy,suggestions=suggestions ,ImageDisplay="http://127.0.0.1:5000/static/images/"+fileName,ImageDisplay1="http://127.0.0.1:5000/static/gray.jpg", ImageDisplay2="http://127.0.0.1:5000/static/threshold.jpg",ImageDisplay3="http://127.0.0.1:5000/static/matrix.png")
    
        return render_template('home.html')
if __name__ == '__main__':
    app.run(debug=True)
