# Author: Thiago Gonçalves Claro dos Santos <thiago.gsantos03@gmail.com>
# Github e Twitter: @tgcsantos

import pickle
from gensim.models import KeyedVectors
from flask import Flask, request, render_template

from utils import tokenizador, combinacao_de_vetores_por_soma

app = Flask(__name__, template_folder="templates")

@app.before_first_request
def load_models():
    
    global classificador
    global w2v_modelo

    w2v_dir = "models/modelo_skipgram.txt"
    classificador_dir = "models/rl_sg.pkl"
    w2v_modelo = KeyedVectors.load_word2vec_format(w2v_dir)
 
    with open(classificador_dir, "rb") as f:
        classificador = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    
    titulo = next(request.form.values())
    
    titulo_tokens = tokenizador(titulo)
    titulo_vetor = combinacao_de_vetores_por_soma(titulo_tokens, w2v_modelo)
    titulo_categoria = classificador.predict(titulo_vetor)

    output = titulo_categoria[0].capitalize()

    return render_template('index.html',
                            title='Título: {}'.format(titulo), 
                            category='Categoria: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
