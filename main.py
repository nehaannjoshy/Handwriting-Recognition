
import numpy as np
from Utils import *
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.models import model_from_json
import shutil
from keras import backend as K
from keras.utils import plot_model
from Spell import correction_list
import tensorflow as tf
from flask import Flask, request, render_template, redirect,send_file
import os
from PIL import Image  
import PIL
from page_to_lines import *

app = Flask(__name__, template_folder='Path_to_Project//Templates')
UPLOAD_FOLDER = 'Path_to_Project//upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
K.clear_session()
with open('Resource/line_model_predict.json', 'r') as f:
		l_model_predict = model_from_json(f.read())
l_model_predict.load_weights('Resource/low_loss.h5')
                             

global graph
graph = tf.get_default_graph()

@app.route("/")
def home():
    return render_template("hr.html")
@app.route("/converted",methods=["POST"])
def upload():
   if request.method == 'POST':
         f = request.files['filen']
         #type(f)
         fname = f.name
         if fname == '':
              return redirect(url_for(home))
         else:
              im1 = Image.open(f)
              #pth = os.path.join(app.config['UPLOAD_FOLDER'], fname) + ".png"
              im1.save(os.path.join(app.config['UPLOAD_FOLDER'], fname) + ".png")
   test_img = 'upload/filen.png'
   f1 = open(fname+ ".txt","w")
   f1.close()
    
   #img = prepareImg(cv2.imread(test_img),255) 
   length = crop(test_img)
   with graph.as_default():
    for i in range(length):   
     pred = predict_image(l_model_predict,"temp/{}.png".format(str(i)) , False)
     f2 = open(fname + ".txt", "a")
     f2.write(pred)
     f2.close()
   #shutil.rmtree('tmp') 
   pth = os.getcwd() 
   f2n = f2.name
   
   f2n = pth + "/" + f2n
   global path
   path= f2n
   #return send_file(path, as_attachment=True)
   return render_template("dwn.html", f2n = f2n)
    
@app.route('/download')
def download_file():
	return send_file(path, as_attachment=True)
if __name__ == "__main__":  
	app.run()        
