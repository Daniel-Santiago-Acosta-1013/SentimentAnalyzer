 # Análisis de Sentimientos con Naive Bayes

 Este proyecto proporciona una interfaz gráfica de usuario (GUI) para entrenar y utilizar un modelo de análisis de sentimientos basado en el algoritmo de Naive Bayes.

 ## Requisitos

 - Python 3.6 o superior.
 - tkinter
 - pandas
 - scikit-learn

 ## Características

 - **Interfaz gráfica fácil de usar**: Este programa proporciona una interfaz gráfica sencilla para entrenar y probar el modelo de análisis de sentimientos.
 - **Compatibilidad con varios formatos de archivo**: El programa puede leer archivos CSV, Excel (XLSX) y JSON para el entrenamiento del modelo.
 - **Algoritmo de Naive Bayes**: Se utiliza el algoritmo de Naive Bayes para el análisis de sentimientos debido a su eficacia y rapidez.

 ## Cómo usar

 1. Ejecute el programa.
 2. Haga clic en "Load CSV" y elija un archivo de datos para entrenar el modelo. El archivo debe tener dos columnas: "text" para el texto del que se extrae el sentimiento y "sentiment" para el sentimiento real (positivo o negativo).
 3. Una vez entrenado el modelo, puedes probarlo ingresando texto en la segunda pestaña y haciendo clic en "Predict Sentiment".

 ## Advertencias

 Antes de usar este programa, tenga en cuenta que el algoritmo de Naive Bayes hace suposiciones muy simplificadas sobre los datos, lo que puede no ser adecuado para todos los conjuntos de datos. Siempre es recomendable experimentar con diferentes algoritmos y ajustes para obtener los mejores resultados.
