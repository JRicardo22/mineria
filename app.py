import sys
from flask import Flask, render_template, request, jsonify,flash,request,url_for,redirect
from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import plot_tree, DecisionTreeClassifier
from werkzeug.utils import secure_filename
import os
import pandas as pd
import io
import base64
import seaborn as sns
import numpy as np
from Modelos.Knn import modelo_knn
from Modelos.Arbol import modelo_arbol
from Modelos.random_forest import modelo_random_forest
from Modelos.adaboost import modelo_adaboost
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from joblib import Parallel, delayed



app = Flask(__name__)
# Configuración de la app
app.secret_key = "Rms"
app.config['CARPETA_SUBIDAS'] = os.path.join(os.getcwd(), 'archivos')  # Ruta absoluta a carpeta
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB máximo

EXTENSIONES_PERMITIDAS = {'csv'}

# Verifica si la extensión es válida
def es_archivo_valido(nombre_archivo):
    return '.' in nombre_archivo and nombre_archivo.rsplit('.', 1)[1].lower() in EXTENSIONES_PERMITIDAS

# Crear carpeta si no existe
os.makedirs(app.config['CARPETA_SUBIDAS'], exist_ok=True)

@app.route('/')
def inicio():
    ruta = app.config['CARPETA_SUBIDAS']
    archivos = [f for f in os.listdir(ruta) if f.endswith('.csv')]
    return render_template('index.html', archivos=archivos)

@app.route('/subir', methods=['POST'])
def cargador_csv():
    if 'archivo' not in request.files:
        flash('No se encontró ningún archivo en la solicitud.', 'danger')
        return redirect(request.referrer)

    archivo = request.files['archivo']

    if archivo.filename == '':
        flash('No se seleccionó ningún archivo.', 'danger')
        return redirect(request.referrer)

    if archivo and es_archivo_valido(archivo.filename):
        nombre_archivo_limpio = secure_filename(archivo.filename)
        ruta_archivo = os.path.join(app.config['CARPETA_SUBIDAS'], nombre_archivo_limpio)
        archivo.save(ruta_archivo)

        flash('Archivo subido correctamente.', 'success')
        return redirect(url_for('lee_archivo', nombre_archivo=nombre_archivo_limpio))
    else:
        flash('Tipo de archivo no permitido. Solo se permiten archivos .csv', 'danger')
        return redirect(request.referrer)


@app.route('/leer_archivo/<nombre_archivo>', methods=['GET'])
def lee_archivo(nombre_archivo):
    ruta_completa = os.path.join(app.config['CARPETA_SUBIDAS'], nombre_archivo)
    try:
        df = pd.read_csv(ruta_completa)
        return render_template('informacion.html', nombre_archivo=nombre_archivo, datos=df.to_html(classes='table table-striped'))
    except Exception as e:
        flash(f"Ocurrió un error al leer el archivo: {e}", 'danger')
        return redirect(url_for('inicio'))

@app.route('/python_editor/<nombre_archivo>')
def pythonEditor(nombre_archivo):
    return render_template('editorP.html', nombre_archivo=nombre_archivo)

@app.route('/run_code', methods=['POST'])
def run_code():
    try:
        # Obtener el código y el nombre del archivo desde el formulario
        user_code = request.form['code']
        nombre_archivo = request.form.get('nombre_archivo')  # <-- CORREGIDO

        if not nombre_archivo:
            return jsonify({'output': 'Error: nombre_archivo no proporcionado.', 'plot_url': None})

        # Cargar el archivo CSV desde la carpeta de subidas
        ruta_archivo = os.path.join(app.config['CARPETA_SUBIDAS'], nombre_archivo)
        df = pd.read_csv(ruta_archivo)

        # Preparar para capturar el output
        old_stdout = sys.stdout
        mystdout = io.StringIO()
        sys.stdout = mystdout

        # Limpiar cualquier gráfica anterior
        plt.clf()

        # Definir el contexto seguro en el cual ejecutar el código
        contexto = {
            'plt': plt,
            'pd': pd,
            'sns': sns,
            'np': np,
            'df': df,
            'train_test_split': train_test_split,
            'RandomizedSearchCV': RandomizedSearchCV,
            'GridSearchCV': GridSearchCV,
            'KNeighborsClassifier': KNeighborsClassifier,
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'plot_tree': plot_tree,
            'RandomForestClassifier': RandomForestClassifier,
            'AdaBoostClassifier': AdaBoostClassifier,
            'accuracy_score': accuracy_score,
            'precision_score': precision_score,
            'recall_score': recall_score,
            'f1_score': f1_score,
            'classification_report': classification_report,
            'confusion_matrix': confusion_matrix,
            'mutual_info_classif': mutual_info_classif,
        }

        # Ejecutar el código del usuario en el contexto dado
        exec(user_code, contexto)

        # Capturar cualquier salida generada
        output = mystdout.getvalue()

        # Verificar si se generó un gráfico
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

    except Exception as e:
        output = f"Error: {str(e)}"
        plot_url = None
    finally:
        sys.stdout = old_stdout  # Restaurar la salida estándar

    return jsonify({'output': output, 'plot_url': plot_url})

@app.route('/consultar/<nombre_archivo>', methods=['GET', 'POST'])
def consultar(nombre_archivo):
    ruta_completa = os.path.join(app.config['CARPETA_SUBIDAS'], nombre_archivo)

    try:
        # Cargar el DataFrame
        df = pd.read_csv(ruta_completa)

        # Procesar el formulario si se envía
        if request.method == 'POST':
            consulta = request.form['consulta']
            resultado = None

            # 1. Promedio de las columnas numéricas
            if consulta == 'promedio':
                resultado = df.mean()

            # 2. Filtro por valor específico en la columna 'Level'
            elif consulta == 'filtro':
                nivel = request.form.get('nivel')  # Obtenemos el valor de nivel desde el formulario
                if nivel:
                    resultado = df[df['Level'] == nivel]

            # 3. Estadísticas descriptivas
            elif consulta == 'estadisticas':
                resultado = df.describe()

            # 4. Contar ocurrencias en una columna
            elif consulta == 'contar_ocurrencias':
                columna = request.form.get('columna')  # Obtenemos el nombre de la columna
                if columna in df.columns:
                    resultado = df[columna].value_counts()

            # 5. Filtro por rangos numéricos (Ejemplo: Filtrar por edad)
            elif consulta == 'filtro_rango':
                columna = request.form.get('columna_rango')  # Columna para rango
                minimo = float(request.form.get('minimo'))  # Valor mínimo
                maximo = float(request.form.get('maximo'))  # Valor máximo
                if columna in df.columns:
                    resultado = df[(df[columna] >= minimo) & (df[columna] <= maximo)]

            # Convertir el resultado en una tabla HTML
            if isinstance(resultado, pd.DataFrame) or isinstance(resultado, pd.Series):
                resultado_html = resultado.to_html(classes='table table-bordered table-hover')
            else:
                resultado_html = f"<p>{resultado}</p>"

            return render_template('consultas.html', nombre_archivo=nombre_archivo, resultado_html=resultado_html)

        # Si es GET, solo mostrar el formulario
        return render_template('consultas.html', nombre_archivo=nombre_archivo)

    except Exception as e:
        flash(f"Ocurrió un error al procesar la consulta: {e}", 'danger')
        return redirect(url_for('inicio'))


@app.route('/resultados/<nombre_archivo>', methods=['GET'])
def resultados(nombre_archivo):
    try:
        ruta_completa = os.path.join(app.config['CARPETA_SUBIDAS'], nombre_archivo)
        df = pd.read_csv(ruta_completa)

        resultados = []

        _, acc_knn, report_knn, cm_knn = modelo_knn(df.copy())
        _, acc_arbol, report_arbol, cm_arbol = modelo_arbol(df.copy())
        _, acc_random, report_random, cm_random = modelo_random_forest(df.copy())
        _, acc_ada, report_ada, cm_ada = modelo_adaboost(df.copy())

        resultados.append({
            'nombre': 'KNN',
            'accuracy': acc_knn,
            'reporte': report_knn,
            'confusion': cm_knn.tolist()
        })

        resultados.append({
            'nombre': 'Árbol de Decisión',
            'accuracy': acc_arbol,
            'reporte': report_arbol,
            'confusion': cm_arbol.tolist()
        })

        resultados.append({
            'nombre': 'Random Forest',
            'accuracy': acc_random,
            'reporte': report_random,
            'confusion': cm_random.tolist()
        })

        resultados.append({
            'nombre': 'AdaBoost',
            'accuracy': acc_ada,
            'reporte': report_ada,
            'confusion': cm_ada.tolist()
        })

        return render_template('Resultados.html', resultados=resultados)

    except Exception as e:
        return f"Ha ocurrido un error al procesar el archivo: {str(e)}", 500



@app.route('/Metodologia')
def Metodologia():
    return render_template('Metodologia.html')


@app.route('/integrantes')
def integrantes():
    return render_template('integrantes.html')


@app.route('/analizar')
def analizar():
    return "hola"

if __name__ == '__main__':
    app.run()


@app.route('/graficas/<nombre_archivo>')
def mostrar_graficas(nombre_archivo):
    ruta_csv = os.path.join(app.config['CARPETA_SUBIDAS'], nombre_archivo)
    try:
        df = pd.read_csv(ruta_csv)

        if 'Level' not in df.columns:
            flash("El archivo no contiene la columna 'Level'", 'danger')
            return redirect(url_for('inicio'))

        ruta_salida = os.path.join('static', 'graficas')
        os.makedirs(ruta_salida, exist_ok=True)

        # --- Gráfica de pastel ---
        tdf = df['Level'].value_counts().reset_index()
        tdf.columns = ['Level', 'count']
        plt.figure(figsize=(6, 4))
        plt.pie(tdf['count'], labels=tdf['Level'], autopct='%.2f%%')
        plt.title('Distribución por Nivel')
        ruta_pie = os.path.join(ruta_salida, f'pie_{nombre_archivo}.png')
        plt.savefig(ruta_pie)
        plt.close()

        # --- Histograma de la primera columna numérica ---
        col_numericas = df.select_dtypes(include='number').columns
        if len(col_numericas) == 0:
            flash("El archivo no tiene columnas numéricas para graficar", 'danger')
            return redirect(url_for('inicio'))

        plt.figure(figsize=(6, 4))
        df[col_numericas[0]].hist(bins=20, color='skyblue', edgecolor='black')
        plt.title(f'Histograma de {col_numericas[0]}')
        plt.xlabel(col_numericas[0])
        plt.ylabel('Frecuencia')
        ruta_hist = os.path.join(ruta_salida, f'hist_{nombre_archivo}.png')
        plt.savefig(ruta_hist)
        plt.close()

        # --- Heatmap de correlaciones ---
        columnas_corr = ['Air Pollution', 'Alcohol use', 'Dust Allergy', 'Genetic Risk',
                         'chronic Lung Disease', 'Balanced Diet', 'Obesity', 'Smoking']

        columnas_existentes = [col for col in columnas_corr if col in df.columns]

        if len(columnas_existentes) >= 2:
            corr = df[columnas_existentes].corr()
            plt.figure(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Mapa de Calor de Riesgos de Salud')
            ruta_heatmap = os.path.join(ruta_salida, f'heatmap_{nombre_archivo}.png')
            plt.tight_layout()
            plt.savefig(ruta_heatmap)
            plt.close()
        else:
            ruta_heatmap = None

        # --- Gráficos combinados (histogramas y conteos) ---
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))

        sns.histplot(df['Age'], bins=10, kde=True, ax=axes[0, 0])
        axes[0, 0].set_title('Histograma de Edad')

        sns.histplot(df['Air Pollution'], bins=10, kde=True, ax=axes[0, 1])
        axes[0, 1].set_title('Histograma de Contaminación del Aire')

        sns.histplot(df['Alcohol use'], bins=2, kde=True, ax=axes[1, 0])
        axes[1, 0].set_title('Histograma de Consumo de Alcohol')

        sns.countplot(x='Gender', data=df, ax=axes[1, 1])
        axes[1, 1].set_title('Conteo por Género')

        sns.countplot(x='Dust Allergy', data=df, ax=axes[2, 0])
        axes[2, 0].set_title('Conteo por Alergia al Polvo')

        sns.countplot(x='Smoking', data=df, ax=axes[2, 1])
        axes[2, 1].set_title('Conteo por Tabaquismo')

        plt.tight_layout()
        ruta_combinada = os.path.join(ruta_salida, f'combinada_{nombre_archivo}.png')
        plt.savefig(ruta_combinada)
        plt.close()

        return render_template('graficas.html',
                               nombre_archivo=nombre_archivo,
                               ruta_pie=url_for('static', filename=f'graficas/pie_{nombre_archivo}.png'),
                               ruta_hist=url_for('static', filename=f'graficas/hist_{nombre_archivo}.png'),
                               ruta_heatmap=url_for('static',
                                                    filename=f'graficas/heatmap_{nombre_archivo}.png') if ruta_heatmap else None,
                               ruta_combinada=url_for('static', filename=f'graficas/combinada_{nombre_archivo}.png')
                               )




    except Exception as e:
        flash(f"Ocurrió un error al generar las gráficas: {e}", 'danger')
        return redirect(url_for('inicio'))

