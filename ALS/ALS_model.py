import itertools
import os
import sys
from pyspark.ml.evaluation import RankingEvaluator, RegressionEvaluator
from pyspark.ml.recommendation import ALS 
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml.feature import BucketedRandomProjectionLSH 
from pyspark.sql.functions import col, expr,udf
from pyspark.sql import SparkSession 
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import log1p
from pyspark.sql.functions import count
from pyspark.ml.recommendation import ALSModel

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

env_root = os.path.dirname(sys.executable)

# En Windows con Conda, los DLLs de aceleración viven en 'Library\bin'
library_bin = os.path.join(env_root, "Library", "bin")

# Verificamos si existe y lo inyectamos al principio del PATH del sistema
if os.path.exists(library_bin):
    print(f"--> Inyectando librerías nativas desde: {library_bin}")
    # Ponerlo al PRINCIPIO asegura que Windows mire aquí antes que en System32
    os.environ['PATH'] = library_bin + os.pathsep + os.environ['PATH']
else:
    print("--> ADVERTENCIA: No se encontró la carpeta Library/bin")

spark = SparkSession.builder \
    .appName("ALS") \
    .config("spark.master", "local[2]") \
    .config("spark.driver.memory", "10g") \
    .getOrCreate()

spark.sparkContext.setCheckpointDir("checkpoints_als")

data = spark.read.csv(
    "musicDB.csv",
    header=True,
    inferSchema=True,
)

user_counts = data.groupBy("user_id").count().withColumnRenamed("count", "user_n")
song_counts = data.groupBy("song_id").count().withColumnRenamed("count", "song_n")

# 2. Filtramos: Solo usuarios con al menos 5 canciones, y canciones escuchadas min 5 veces
# (Puedes ser más agresivo y poner 10 si quieres subir más el score)
data_clean = data.join(user_counts, "user_id").join(song_counts, "song_id") \
                 .filter((col("user_n") >= 5) & (col("song_n") >= 5))

print(f"Datos originales: {data.count()}")
print(f"Datos limpios: {data_clean.count()}")

# Reemplazamos la variable data
data = data_clean

data = data.withColumn("rating_log", log1p(col("play_count")))

(training, test ) = data.randomSplit([0.8, 0.2])

training.cache()

print(f"Datos de entrenamiento cacheados: {training.count()} filas.")

# 1. Define tus rangos de parámetros manualmente
ranks = [150]       # Prueba valores bajos y medios
regParams = [0.1]  # La regularización es clave
alphas = [1.0]# La confianza en los datos
maxIters = [20]

# Guardaremos el mejor modelo
best_ndcg = 0.0
best_params = {}
best_model_manual = None


# 2. Pre-procesa el set de Test para tener los "items_true" listos (ahorra tiempo)
# Agrupamos una sola vez lo que el usuario REALMENTE escuchó en test
test_ground_truth = test.groupBy("user_id").agg(
    expr("collect_set(cast(song_id as double))").alias("items_true")
)

test_ground_truth.cache()

print(f"Usuarios de test cacheados: {test_ground_truth.count()}")

print("Iniciando Grid Search Manual...")
print("-" * 50)

evaluator_RE = RankingEvaluator(
    metricName="ndcgAtK",  
    k=10,                       
    labelCol="items_true",      
    predictionCol="prediction"  
)

# 3. Iteramos sobre todas las combinaciones
for rank, reg, alpha, iterations in itertools.product(ranks, regParams, alphas, maxIters):
    
    # A. Entrenar
    als = ALS(
        rank=rank, 
        maxIter=iterations, 
        regParam=reg, 
        alpha=alpha,
        userCol='user_id', 
        itemCol='song_id', 
        ratingCol='rating_log', 
        coldStartStrategy='drop', 
        implicitPrefs=True, 
        nonnegative=True,
        checkpointInterval=5
    )
    
    model = als.fit(training)
    
    # B. Predecir / Generar Recomendaciones Top-K
    recs = model.recommendForAllUsers(10)
    
    # C. Formatear predicciones para el evaluador
    recs_formatted = recs.select(
        col("user_id"),
        expr("transform(recommendations, x -> double(x.song_id))").alias("prediction")
    )
    
    # D. Unir con la verdad (Ground Truth)
    # Hacemos join con el test set ya agrupado
    eval_df = recs_formatted.join(test_ground_truth, on="user_id", how="inner")
    
    # E. Calcular nDCG
    current_ndcg = evaluator_RE.evaluate(eval_df)
    
    print(f"Params: Rank={rank}, Reg={reg}, Alpha={alpha} --> nDCG={current_ndcg:.4f}")
    
    # F. Guardar si es el mejor
    if current_ndcg > best_ndcg:
        best_ndcg = current_ndcg
        best_model_manual = model
        best_params = {"rank": rank, "reg": reg, "alpha": alpha}
print(f"Datos de entrenamiento cacheados: {training.count()} filas.")
print("-" * 50)
print(f"¡Mejor nDCG encontrado: {best_ndcg:.4f}!")
print(f"Con parámetros: {best_params}")

save_path = "./modelo_entrenado"
best_model_manual.save(save_path)

