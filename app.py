from flask import render_template, redirect
from icfes_api import app

# Rutas de páginas (Front)
@app.route('/')
@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/dashboard')
def dashboard_page():
    return render_template('dashboard.html')

@app.route('/simulator')
def simulator_page():
    return render_template('icfes_exam_simulator.html')

@app.route('/generator')
def generator_page():
    return render_template('icfes_generator_mcq.html')

@app.route('/pdf-evaluation')
def pdf_evaluation_page():
    return render_template('icfes_pdf_evaluation.html')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
