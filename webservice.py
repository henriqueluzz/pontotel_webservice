# -*- coding: utf-8 -*-

from flask import Flask,send_file
from io import BytesIO
from PIL import Image
import imutils
import numpy as np
import flask
import cv2
import os

'''
Para testar: jogar no console
curl -X POST -F file=@20180309_165436.jpg 'http://127.0.0.1:5000/predict'>out.jpeg
curl -X POST -F file=@20180309_165436.jpg 'https://pontotel-image.herokuapp.com/predict'>out.jpeg

'''
app = Flask(__name__)

@app.route("/")
def index():
    return "<h1>It works</h1>"

@app.route("/predict", methods=['POST'])
def predict():
    
    #Le a imagem que foi enviada ao servidor
    img = flask.request.files['file'].read()
    
    #Cria uma ndarray apartir do buffer
    nparr = np.frombuffer(img, np.uint8)
    
    #faz o decode da imagem
    img = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
    
    #faz a transformação e retorna a imagem
    width = img.shape[1]	
    height = img.shape[0]	 
        
    r = 500/width
    dim = (500,int(r*height))
        
    #Aplica um resize na imagem mantenro o ratio dela
    img = cv2.resize(img,dim,interpolation = cv2.INTER_AREA) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    #Aplcia um Blur Gaussiano com kernel tamanho 5x5
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
        
    #Aplica a função findCOntours para encontrar os contornos da mesa de xadrex
    
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
    
    '''
    Dentre os contornos encontrados, retorna o de maior área
    '''
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True) 
        if len(approx) == 4:
                screenCnt = approx
                break
    
    '''
    Ordena os 4 pares de coordenadas
    '''    
    pts = screenCnt.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = "float32")
    
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
     	[0, 0],
        	[maxWidth - 1, 0],
        	[maxWidth - 1, maxHeight - 1],
        	[0, maxHeight - 1]], dtype = "float32")
    
    '''
    Aplica a PerspectiveTransform baseado nos 4 pontos encontrados 
    '''
    M = cv2.getPerspectiveTransform(rect, dst)
    warped_image = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    
    #cria uma imagem PIL
    image  = Image.fromarray(warped_image.astype('uint8'),'RGB')
    
    output = BytesIO()
    #Streama a imagem em bytes pro cliente 
    image.save(output, format = 'JPEG',quality = 70)
    output.seek(0)
    
    return send_file(output,mimetype = 'image/jpeg')
    
if __name__ == "__main__":
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host = "0.0.0.0", port = port)
