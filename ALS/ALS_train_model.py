from pyspark.ml.recommendation import ALSModel
from pyspark.sql.functions import udf, col
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.sql import SparkSession
import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

spark = SparkSession.builder.getOrCreate()

loaded_model = ALSModel.load("./modelo_entrenado")
print(f"Modelo cargado exitosamente desde {loaded_model}")


def recomendar_usuarios_similares(als_model, user_id, spark, top_k=10, path_usuarios=None):
    """
    Dado un modelo entrenado ALS y un user_id,
    recomienda otros usuarios similares basados en los vectores latentes del modelo.
    """

    #Obtener los factores latentes del modelo ALS
    user_features = als_model.userFactors

    #Convertir la columna 'features' (array) a vector
    array_to_vector_udf = udf(lambda x: Vectors.dense(x), VectorUDT())
    user_features_vec = user_features.withColumn("features_vec", array_to_vector_udf("features"))

    #Crear un modelo de similitud basado en LSH (Locality-Sensitive Hashing)
    lsh = BucketedRandomProjectionLSH(
        inputCol="features_vec",
        outputCol="hashes",
        numHashTables=5,
        bucketLength=2.0
    ).fit(user_features_vec)

    #Buscar el vector del usuario solicitado
    try:
        key = (user_features_vec
               .filter(col("id") == user_id)
               .select("features_vec")
               .head()
               .features_vec)
    except:
        return f"El user_id {user_id} NO existe en el modelo."

    #Buscar los usuarios más cercanos
    similar_users = lsh.approxNearestNeighbors(
        dataset=user_features_vec,
        key=key,
        numNearestNeighbors=top_k + 1  # +1 porque incluye al mismo usuario
    )

    #Filtrar al usuario original
    resultado = similar_users.filter(col("id") != user_id)

    resultado.select("id", "distCol")

    if path_usuarios:
        usuarios_df = spark.read.csv(path_usuarios, header=True, inferSchema=True)
        resultado = resultado.withColumnRenamed("id", "user_id")
        resultado = resultado.join(usuarios_df, on="user_id", how="left")
    
    return resultado.select("user_id", "distCol", "Username") if path_usuarios else resultado



result = recomendar_usuarios_similares(als_model=loaded_model,user_id=1,spark=spark,top_k=10, path_usuarios="usuarios_completos.csv")

result.show(truncate=False)