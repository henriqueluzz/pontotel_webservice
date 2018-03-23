Para rodar o webservice localmente:

pip install -r reuquirements.txt

python webservice.py

Para testar o webservice:
curl -X POST -F file=@imagm.jpg 'http://localhost:5000/predict'>out.jpeg

A imagem transformada sera salva como out.jpeg.





