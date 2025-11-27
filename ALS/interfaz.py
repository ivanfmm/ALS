# -*- coding: utf-8 -*-
"""
Sistema de Recomendación de Usuarios - Interfaz Gráfica
Basado en ALS (Alternating Least Squares) con PySpark
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import os
import sys
from threading import Thread
import queue

# Configuración de PySpark
os.environ["JAVA_HOME"] = r"C:/Program Files/Eclipse Adoptium/jdk-21.0.5.11-hotspot"
os.environ["HADOOP_HOME"] = "C:/hadoop"
os.environ["PATH"] = os.environ["HADOOP_HOME"] + "/bin;" + os.environ["PATH"]
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


class SistemaRecomendacionUsuarios:
    def __init__(self):
        self.spark = None
        self.modelo = None
        self.modelo_cargado = False
        self.db_conectada = False
        self.usuario_actual = None
        
    def verificar_conexiones(self):
        """Verifica el estado del modelo y la base de datos"""
        try:
            # Verificar modelo
            model_path = "modelo_als"
            if os.path.exists(model_path):
                self.modelo_cargado = True
            else:
                self.modelo_cargado = False
            
            # Verificar datos
            if os.path.exists("usuarios_completos.csv"):
                self.db_conectada = True
            else:
                self.db_conectada = False
                
            return True
        except Exception as e:
            print(f"Error verificando conexiones: {e}")
            return False
    
    def cargar_modelo(self):
        """Carga el modelo ALS entrenado"""
        try:
            from pyspark.ml.recommendation import ALSModel
            from pyspark.sql import SparkSession
            
            # Iniciar Spark
            self.spark = SparkSession.builder \
                .appName("RecomendacionUsuarios") \
                .config("spark.master", "local[*]") \
                .config("spark.driver.memory", "8g") \
                .getOrCreate()
            
            # Cargar modelo
            model_path = "modelo_als"
            self.modelo = ALSModel.load(model_path)
            self.modelo_cargado = True
            
            return True, "Modelo cargado exitosamente"
        except Exception as e:
            return False, f"Error cargando modelo: {str(e)}"
    
    def recomendar_usuarios_similares(self, user_id, top_k=10):
        """
        Recomienda usuarios similares basándose en los vectores latentes del modelo ALS
        """
        try:
            from pyspark.sql.functions import udf, col
            from pyspark.ml.linalg import Vectors, VectorUDT
            from pyspark.ml.feature import BucketedRandomProjectionLSH
            
            # Obtener factores latentes
            user_features = self.modelo.userFactors
            
            # Convertir array a vector
            array_to_vector_udf = udf(lambda x: Vectors.dense(x), VectorUDT())
            user_features_vec = user_features.withColumn(
                "features_vec", 
                array_to_vector_udf("features")
            )
            
            # Crear modelo LSH para similitud
            lsh = BucketedRandomProjectionLSH(
                inputCol="features_vec",
                outputCol="hashes",
                numHashTables=5,
                bucketLength=2.0
            ).fit(user_features_vec)
            
            # Buscar vector del usuario
            try:
                key = (user_features_vec
                       .filter(col("id") == user_id)
                       .select("features_vec")
                       .head()
                       .features_vec)
            except:
                return None, f"El usuario {user_id} no existe en el modelo"
            
            # Buscar usuarios más cercanos
            similar_users = lsh.approxNearestNeighbors(
                dataset=user_features_vec,
                key=key,
                numNearestNeighbors=top_k + 1
            )
            
            # Filtrar al usuario original
            resultado = similar_users.filter(col("id") != user_id)
            
            # Cargar información de usuarios si existe
            if os.path.exists("usuarios_completos.csv"):
                usuarios_df = self.spark.read.csv(
                    "usuarios_completos.csv", 
                    header=True, 
                    inferSchema=True
                )
                resultado = resultado.withColumnRenamed("id", "user_id")
                resultado = resultado.join(usuarios_df, on="user_id", how="left")
                
                # Recolectar resultados
                resultados_lista = resultado.select(
                    "user_id", "distCol", "Username"
                ).collect()
                
                return resultados_lista, None
            else:
                # Sin información adicional
                resultados_lista = resultado.select("id", "distCol").collect()
                return resultados_lista, None
                
        except Exception as e:
            return None, f"Error en recomendación: {str(e)}"
    
    def obtener_canciones_usuario(self, user_id, limit=50):
        """
        Obtiene las canciones escuchadas por un usuario específico
        con información de canción y artista
        """
        try:
            from pyspark.sql.functions import col
            
            # Cargar datos de música
            if not os.path.exists("musicDB.csv"):
                return None, "No se encontró el archivo musicDB.csv"
            
            music_df = self.spark.read.csv(
                "musicDB.csv",
                header=True,
                inferSchema=True
            )
            
            # Filtrar por usuario
            user_songs = music_df.filter(col("user_id") == user_id)
            
            # Verificar si el usuario existe
            if user_songs.count() == 0:
                return None, f"El usuario {user_id} no tiene canciones registradas"
            
            # Cargar información de canciones si existe
            if os.path.exists("canciones_escuchadas.csv"):
                songs_info = self.spark.read.csv(
                    "canciones_escuchadas.csv",
                    header=True,
                    inferSchema=True
                )
                # Unir con información de canciones
                user_songs = user_songs.join(songs_info, on="song_id", how="left")
            
            # Ordenar por play_count (más escuchadas primero)
            user_songs = user_songs.orderBy(col("play_count").desc())
            
            # Limitar resultados
            if limit:
                user_songs = user_songs.limit(limit)
            
            # Recolectar resultados
            resultados = user_songs.collect()
            
            return resultados, None
            
        except Exception as e:
            return None, f"Error obteniendo canciones: {str(e)}"
    
    def obtener_estadisticas_usuario(self, user_id):
        """
        Obtiene estadísticas del usuario
        """
        try:
            from pyspark.sql.functions import col, sum as spark_sum, count, avg
            
            if not os.path.exists("musicDB.csv"):
                return None, "No se encontró el archivo musicDB.csv"
            
            music_df = self.spark.read.csv(
                "musicDB.csv",
                header=True,
                inferSchema=True
            )
            
            # Filtrar por usuario
            user_songs = music_df.filter(col("user_id") == user_id)
            
            # Calcular estadísticas
            stats = user_songs.agg(
                count("song_id").alias("total_canciones"),
                spark_sum("play_count").alias("total_reproducciones"),
                avg("play_count").alias("promedio_reproducciones")
            ).collect()
            
            if stats:
                return stats[0], None
            else:
                return None, "No se encontraron datos para este usuario"
                
        except Exception as e:
            return None, f"Error obteniendo estadísticas: {str(e)}"


class InterfazSoundTrack:
    def __init__(self, root):
        self.root = root
        self.root.title("SoundTrack - Sistema de Recomendación Musical")
        self.root.geometry("1400x850")
        self.root.configure(bg="#000000")
        
        # Sistema de recomendación
        self.sistema = SistemaRecomendacionUsuarios()
        self.queue = queue.Queue()
        
        # Configurar estilo
        self.configurar_estilos()
        
        # Crear interfaz
        self.crear_header()
        self.crear_pestanas()
        self.crear_footer()
        
        # Verificar estado inicial
        self.verificar_conexiones_inicial()
        
    def configurar_estilos(self):
        """Configura los estilos visuales de la aplicación"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Paleta de colores
        self.colors = {
            'base_black': '#000000',
            'soft_black': '#0A0A0A',
            'electric_blue': '#0080FF',
            'neon_green': '#00FF41',
            'dark_gray': '#1A1A1A',
            'white': '#FFFFFF',
            'cyan': '#00FFD4'
        }
        
        # Estilo para notebook (pestañas)
        style.configure('TNotebook', 
                       background=self.colors['base_black'],
                       borderwidth=0)
        style.configure('TNotebook.Tab',
                       background=self.colors['soft_black'],
                       foreground=self.colors['white'],
                       padding=[20, 10],
                       borderwidth=1)
        style.map('TNotebook.Tab',
                 background=[('selected', self.colors['base_black'])],
                 foreground=[('selected', self.colors['neon_green'])])
        
        # Estilo para frames
        style.configure('Dark.TFrame', background=self.colors['base_black'])
        
        # Estilo para labels
        style.configure('Title.TLabel',
                       background=self.colors['base_black'],
                       foreground=self.colors['neon_green'],
                       font=('Arial', 48, 'bold'))
        style.configure('Subtitle.TLabel',
                       background=self.colors['base_black'],
                       foreground='#888888',
                       font=('Arial', 16))
        style.configure('Status.TLabel',
                       background=self.colors['soft_black'],
                       foreground=self.colors['white'],
                       font=('Arial', 12, 'bold'),
                       padding=10)
        style.configure('StatusTitle.TLabel',
                       background=self.colors['soft_black'],
                       foreground='#888888',
                       font=('Arial', 10))
        
        # Estilo para botones
        style.configure('Green.TButton',
                       background=self.colors['neon_green'],
                       foreground=self.colors['base_black'],
                       font=('Arial', 12, 'bold'),
                       borderwidth=0,
                       padding=[20, 10])
        
    def crear_header(self):
        """Crea el encabezado de la aplicación"""
        header_frame = ttk.Frame(self.root, style='Dark.TFrame')
        header_frame.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        # Logo y título
        title_label = ttk.Label(
            header_frame,
            text="SoundTrack",
            style='Title.TLabel'
        )
        title_label.pack(side=tk.LEFT)
        
        subtitle_label = ttk.Label(
            header_frame,
            text="Music Recommendation System",
            style='Subtitle.TLabel'
        )
        subtitle_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # Usuario actual
        self.user_label = ttk.Label(
            header_frame,
            text="Sin usuario",
            style='Subtitle.TLabel'
        )
        self.user_label.pack(side=tk.RIGHT)
        
    def crear_pestanas(self):
        """Crea el sistema de pestañas"""
        # Notebook para pestañas
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Pestaña: Inicio
        self.crear_pestana_inicio()
        
        # Pestaña: Login (Selección de Usuario)
        self.crear_pestana_login()
        
        # Pestaña: Historial de Canciones (NUEVA)
        self.crear_pestana_historial()
        
        # Pestaña: Recomendaciones de Usuarios
        self.crear_pestana_recomendaciones()
        
        # Pestaña: Modelo
        self.crear_pestana_modelo()
        
    def crear_pestana_inicio(self):
        """Crea la pestaña de inicio"""
        inicio_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(inicio_frame, text="🏠 Inicio")
        
        # Contenido centrado
        content_frame = ttk.Frame(inicio_frame, style='Dark.TFrame')
        content_frame.place(relx=0.5, rely=0.4, anchor=tk.CENTER)
        
        # Título de bienvenida
        welcome_label = ttk.Label(
            content_frame,
            text="Bienvenido a SoundTrack",
            font=('Arial', 42, 'bold'),
            foreground='#00FF41',
            background='#000000'
        )
        welcome_label.pack(pady=(0, 10))
        
        # Subtítulo
        subtitle_label = ttk.Label(
            content_frame,
            text="Sistema inteligente de recomendación de usuarios similares",
            style='Subtitle.TLabel'
        )
        subtitle_label.pack(pady=(0, 40))
        
        # Contenedor de estados
        status_container = ttk.Frame(content_frame, style='Dark.TFrame')
        status_container.pack(pady=20)
        
        # Estado Base de Datos
        self.crear_card_estado(
            status_container,
            "Base de Datos",
            "db_status",
            0
        )
        
        # Estado Modelo ML
        self.crear_card_estado(
            status_container,
            "Modelo ML",
            "model_status",
            1
        )
        
        # Usuario Activo
        self.crear_card_estado(
            status_container,
            "Usuario Activo",
            "user_status",
            2
        )
        
        # Botón de verificación
        verify_button = tk.Button(
            inicio_frame,
            text="🔄 Verificar Conexiones",
            font=('Arial', 12, 'bold'),
            bg='#00FF41',
            fg='#000000',
            activebackground='#00FFD4',
            activeforeground='#000000',
            relief=tk.FLAT,
            cursor='hand2',
            padx=30,
            pady=15,
            command=self.verificar_conexiones
        )
        verify_button.place(relx=0.5, rely=0.75, anchor=tk.CENTER)
        
    def crear_card_estado(self, parent, titulo, attr_name, columna):
        """Crea una tarjeta de estado"""
        card_frame = tk.Frame(
            parent,
            bg='#0A0A0A',
            highlightbackground='#1A1A1A',
            highlightthickness=2
        )
        card_frame.grid(row=0, column=columna, padx=15, pady=10, sticky='nsew')
        
        # Configurar tamaño mínimo
        card_frame.config(width=280, height=140)
        card_frame.grid_propagate(False)
        
        # Título
        titulo_label = ttk.Label(
            card_frame,
            text=titulo,
            style='StatusTitle.TLabel'
        )
        titulo_label.pack(pady=(15, 5))
        
        # Estado
        status_label = ttk.Label(
            card_frame,
            text="Sin seleccionar" if "Usuario" in titulo else "✗ Error",
            style='Status.TLabel'
        )
        status_label.pack(pady=10)
        
        # Guardar referencia
        setattr(self, attr_name, status_label)
        
    def crear_pestana_login(self):
        """Crea la pestaña de selección de usuario"""
        login_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(login_frame, text="👤 Login")
        
        # Contenido centrado
        content_frame = ttk.Frame(login_frame, style='Dark.TFrame')
        content_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Título
        title_label = ttk.Label(
            content_frame,
            text="Seleccionar Usuario",
            font=('Arial', 32, 'bold'),
            foreground='#00FF41',
            background='#000000'
        )
        title_label.pack(pady=(0, 30))
        
        # Input de usuario
        input_frame = ttk.Frame(content_frame, style='Dark.TFrame')
        input_frame.pack(pady=20)
        
        label = ttk.Label(
            input_frame,
            text="ID de Usuario:",
            font=('Arial', 14),
            foreground='#FFFFFF',
            background='#000000'
        )
        label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.user_id_entry = tk.Entry(
            input_frame,
            font=('Arial', 14),
            width=15,
            bg='#0A0A0A',
            fg='#FFFFFF',
            insertbackground='#00FF41',
            relief=tk.FLAT,
            bd=5
        )
        self.user_id_entry.pack(side=tk.LEFT)
        
        # Botón de login
        login_button = tk.Button(
            content_frame,
            text="Seleccionar Usuario",
            font=('Arial', 12, 'bold'),
            bg='#00FF41',
            fg='#000000',
            activebackground='#00FFD4',
            activeforeground='#000000',
            relief=tk.FLAT,
            cursor='hand2',
            padx=40,
            pady=15,
            command=self.seleccionar_usuario
        )
        login_button.pack(pady=30)
    
    def crear_pestana_historial(self):
        """Crea la pestaña de historial de canciones del usuario"""
        hist_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(hist_frame, text="🎧 Historial")
        
        # Título
        title_label = ttk.Label(
            hist_frame,
            text="Historial de Canciones",
            font=('Arial', 24, 'bold'),
            foreground='#00FF41',
            background='#000000'
        )
        title_label.pack(pady=20)
        
        # Panel de control
        control_frame = ttk.Frame(hist_frame, style='Dark.TFrame')
        control_frame.pack(fill=tk.X, padx=40, pady=10)
        
        # Campo para ID de usuario a consultar
        ttk.Label(
            control_frame,
            text="ID Usuario:",
            font=('Arial', 12),
            foreground='#FFFFFF',
            background='#000000'
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        self.historial_user_entry = tk.Entry(
            control_frame,
            font=('Arial', 12),
            width=10,
            bg='#0A0A0A',
            fg='#FFFFFF',
            insertbackground='#00FF41',
            relief=tk.FLAT,
            bd=5
        )
        self.historial_user_entry.pack(side=tk.LEFT, padx=(0, 20))
        
        # Límite de canciones
        ttk.Label(
            control_frame,
            text="Mostrar canciones:",
            font=('Arial', 12),
            foreground='#FFFFFF',
            background='#000000'
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        self.num_songs = tk.Spinbox(
            control_frame,
            from_=10,
            to=200,
            width=8,
            font=('Arial', 12),
            bg='#0A0A0A',
            fg='#FFFFFF',
            buttonbackground='#1A1A1A',
            relief=tk.FLAT
        )
        self.num_songs.delete(0, tk.END)
        self.num_songs.insert(0, "50")
        self.num_songs.pack(side=tk.LEFT, padx=(0, 20))
        
        # Botón de cargar historial
        hist_button = tk.Button(
            control_frame,
            text="📋 Cargar Historial",
            font=('Arial', 12, 'bold'),
            bg='#0080FF',
            fg='#FFFFFF',
            activebackground='#00FFD4',
            activeforeground='#000000',
            relief=tk.FLAT,
            cursor='hand2',
            padx=20,
            pady=10,
            command=self.cargar_historial
        )
        hist_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Botón de estadísticas
        stats_button = tk.Button(
            control_frame,
            text="📊 Estadísticas",
            font=('Arial', 12, 'bold'),
            bg='#9933FF',
            fg='#FFFFFF',
            activebackground='#00FFD4',
            activeforeground='#000000',
            relief=tk.FLAT,
            cursor='hand2',
            padx=20,
            pady=10,
            command=self.mostrar_estadisticas
        )
        stats_button.pack(side=tk.LEFT)
        
        # Frame para estadísticas rápidas
        self.stats_frame = tk.Frame(hist_frame, bg='#0A0A0A')
        self.stats_frame.pack(fill=tk.X, padx=40, pady=10)
        
        # Labels de estadísticas
        self.stats_total_songs = ttk.Label(
            self.stats_frame,
            text="Total canciones: --",
            font=('Arial', 11),
            foreground='#00FFD4',
            background='#0A0A0A'
        )
        self.stats_total_songs.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.stats_total_plays = ttk.Label(
            self.stats_frame,
            text="Total reproducciones: --",
            font=('Arial', 11),
            foreground='#00FF41',
            background='#0A0A0A'
        )
        self.stats_total_plays.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.stats_avg_plays = ttk.Label(
            self.stats_frame,
            text="Promedio por canción: --",
            font=('Arial', 11),
            foreground='#0080FF',
            background='#0A0A0A'
        )
        self.stats_avg_plays.pack(side=tk.LEFT, padx=20, pady=10)
        
        # Área de resultados
        results_frame = tk.Frame(hist_frame, bg='#000000')
        results_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=20)
        
        # Scrolled text para historial
        self.historial_text = scrolledtext.ScrolledText(
            results_frame,
            font=('Consolas', 10),
            bg='#0A0A0A',
            fg='#FFFFFF',
            insertbackground='#00FF41',
            relief=tk.FLAT,
            wrap=tk.NONE,
            state='disabled'
        )
        self.historial_text.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar horizontal
        h_scroll = ttk.Scrollbar(results_frame, orient=tk.HORIZONTAL, command=self.historial_text.xview)
        h_scroll.pack(fill=tk.X)
        self.historial_text.config(xscrollcommand=h_scroll.set)
        
        # Configurar tags para colores
        self.historial_text.tag_configure('header', foreground='#00FF41', font=('Consolas', 10, 'bold'))
        self.historial_text.tag_configure('song_id', foreground='#0080FF')
        self.historial_text.tag_configure('song_name', foreground='#FFFFFF', font=('Consolas', 10, 'bold'))
        self.historial_text.tag_configure('artist', foreground='#FF6B9D')
        self.historial_text.tag_configure('play_count', foreground='#00FFD4')
        self.historial_text.tag_configure('separator', foreground='#444444')
        self.historial_text.tag_configure('stats', foreground='#9933FF', font=('Consolas', 10, 'bold'))
        
    def crear_pestana_recomendaciones(self):
        """Crea la pestaña de recomendaciones de usuarios"""
        rec_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(rec_frame, text="🎵 Recomendaciones")
        
        # Título
        title_label = ttk.Label(
            rec_frame,
            text="Usuarios Similares",
            font=('Arial', 24, 'bold'),
            foreground='#00FF41',
            background='#000000'
        )
        title_label.pack(pady=20)
        
        # Panel de control
        control_frame = ttk.Frame(rec_frame, style='Dark.TFrame')
        control_frame.pack(fill=tk.X, padx=40, pady=10)
        
        # Número de recomendaciones
        ttk.Label(
            control_frame,
            text="Número de recomendaciones:",
            font=('Arial', 12),
            foreground='#FFFFFF',
            background='#000000'
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        self.num_recs = tk.Spinbox(
            control_frame,
            from_=5,
            to=50,
            width=10,
            font=('Arial', 12),
            bg='#0A0A0A',
            fg='#FFFFFF',
            buttonbackground='#1A1A1A',
            relief=tk.FLAT
        )
        self.num_recs.delete(0, tk.END)
        self.num_recs.insert(0, "10")
        self.num_recs.pack(side=tk.LEFT, padx=(0, 20))
        
        # Botón de generar recomendaciones
        rec_button = tk.Button(
            control_frame,
            text="🔍 Generar Recomendaciones",
            font=('Arial', 12, 'bold'),
            bg='#0080FF',
            fg='#FFFFFF',
            activebackground='#00FFD4',
            activeforeground='#000000',
            relief=tk.FLAT,
            cursor='hand2',
            padx=20,
            pady=10,
            command=self.generar_recomendaciones
        )
        rec_button.pack(side=tk.LEFT)
        
        # Área de resultados
        results_frame = tk.Frame(rec_frame, bg='#000000')
        results_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=20)
        
        # Scrolled text para resultados
        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            font=('Consolas', 11),
            bg='#0A0A0A',
            fg='#FFFFFF',
            insertbackground='#00FF41',
            relief=tk.FLAT,
            wrap=tk.WORD,
            state='disabled'
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Configurar tags para colores
        self.results_text.tag_configure('header', foreground='#00FF41', font=('Consolas', 11, 'bold'))
        self.results_text.tag_configure('user_id', foreground='#0080FF', font=('Consolas', 11, 'bold'))
        self.results_text.tag_configure('username', foreground='#00FFD4')
        self.results_text.tag_configure('distance', foreground='#ff6666')
        
    def crear_pestana_modelo(self):
        """Crea la pestaña de información del modelo"""
        model_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(model_frame, text="⚙️ Modelo")
        
        # Título
        title_label = ttk.Label(
            model_frame,
            text="Información del Modelo",
            font=('Arial', 24, 'bold'),
            foreground='#00FF41',
            background='#000000'
        )
        title_label.pack(pady=20)
        
        # Información del modelo
        info_frame = tk.Frame(model_frame, bg='#0A0A0A')
        info_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=20)
        
        self.model_info_text = scrolledtext.ScrolledText(
            info_frame,
            font=('Consolas', 11),
            bg='#0A0A0A',
            fg='#FFFFFF',
            insertbackground='#00FF41',
            relief=tk.FLAT,
            wrap=tk.WORD,
            state='disabled'
        )
        self.model_info_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Botones
        button_frame = ttk.Frame(model_frame, style='Dark.TFrame')
        button_frame.pack(pady=20)
        
        load_button = tk.Button(
            button_frame,
            text="📥 Cargar Modelo",
            font=('Arial', 12, 'bold'),
            bg='#00FF41',
            fg='#000000',
            activebackground='#00FFD4',
            activeforeground='#000000',
            relief=tk.FLAT,
            cursor='hand2',
            padx=20,
            pady=10,
            command=self.cargar_modelo
        )
        load_button.pack(side=tk.LEFT, padx=10)
        
        info_button = tk.Button(
            button_frame,
            text="ℹ️ Mostrar Info",
            font=('Arial', 12, 'bold'),
            bg='#1A1A1A',
            fg='#FFFFFF',
            activebackground='#0A0A0A',
            activeforeground='#FFFFFF',
            relief=tk.FLAT,
            cursor='hand2',
            padx=20,
            pady=10,
            command=self.mostrar_info_modelo
        )
        info_button.pack(side=tk.LEFT, padx=10)
        
    def crear_footer(self):
        """Crea el pie de página"""
        footer_frame = ttk.Frame(self.root, style='Dark.TFrame')
        footer_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
        
        footer_label = ttk.Label(
            footer_frame,
            text="Sistema de Recomendación Musical con PySpark ALS | v1.2",
            font=('Arial', 9),
            foreground='#666666',
            background='#000000'
        )
        footer_label.pack(side=tk.LEFT)
        
    def verificar_conexiones_inicial(self):
        """Verifica las conexiones al iniciar"""
        Thread(target=self._verificar_conexiones_thread, daemon=True).start()
        
    def _verificar_conexiones_thread(self):
        """Thread para verificar conexiones"""
        self.sistema.verificar_conexiones()
        self.root.after(100, self.actualizar_estados)
        
    def verificar_conexiones(self):
        """Verifica las conexiones manualmente"""
        Thread(target=self._verificar_conexiones_thread, daemon=True).start()
        
    def actualizar_estados(self):
        """Actualiza los estados en la interfaz"""
        # Estado de base de datos
        if self.sistema.db_conectada:
            self.db_status.config(
                text="✓ Disponible",
                foreground='#00FF41'
            )
        else:
            self.db_status.config(
                text="✗ Error",
                foreground='#ff6666'
            )
        
        # Estado del modelo
        if self.sistema.modelo_cargado:
            self.model_status.config(
                text="✓ Disponible",
                foreground='#00FF41'
            )
        else:
            self.model_status.config(
                text="✗ No cargado",
                foreground='#ff6666'
            )
        
        # Usuario activo
        if self.sistema.usuario_actual:
            self.user_status.config(
                text=f"Usuario {self.sistema.usuario_actual}",
                foreground='#00FFD4'
            )
            self.user_label.config(text=f"Usuario: {self.sistema.usuario_actual}")
        else:
            self.user_status.config(
                text="Sin seleccionar",
                foreground='#888888'
            )
            
    def seleccionar_usuario(self):
        """Selecciona un usuario"""
        try:
            user_id = int(self.user_id_entry.get())
            self.sistema.usuario_actual = user_id
            self.actualizar_estados()
            messagebox.showinfo(
                "Usuario Seleccionado",
                f"Usuario {user_id} seleccionado correctamente"
            )
        except ValueError:
            messagebox.showerror(
                "Error",
                "Por favor ingresa un ID de usuario válido"
            )
            
    def cargar_modelo(self):
        """Carga el modelo ALS"""
        def _cargar():
            exito, mensaje = self.sistema.cargar_modelo()
            self.root.after(0, lambda: self._mostrar_resultado_carga(exito, mensaje))
            
        Thread(target=_cargar, daemon=True).start()
        
        # Mostrar mensaje de carga
        self.model_info_text.config(state='normal')
        self.model_info_text.delete(1.0, tk.END)
        self.model_info_text.insert(tk.END, "Cargando modelo...\n")
        self.model_info_text.config(state='disabled')
        
    def _mostrar_resultado_carga(self, exito, mensaje):
        """Muestra el resultado de la carga del modelo"""
        self.actualizar_estados()
        if exito:
            messagebox.showinfo("Éxito", mensaje)
            self.mostrar_info_modelo()
        else:
            messagebox.showerror("Error", mensaje)
            
    def mostrar_info_modelo(self):
        """Muestra información del modelo"""
        self.model_info_text.config(state='normal')
        self.model_info_text.delete(1.0, tk.END)
        
        if not self.sistema.modelo_cargado or not self.sistema.modelo:
            self.model_info_text.insert(tk.END, "❌ Modelo no cargado\n\n")
            self.model_info_text.insert(tk.END, "Por favor, carga el modelo primero usando el botón 'Cargar Modelo'")
        else:
            info = "="*60 + "\n"
            info += "INFORMACIÓN DEL MODELO ALS\n"
            info += "="*60 + "\n\n"
            info += "✓ Modelo cargado correctamente\n"
            info += "✓ Tipo: Alternating Least Squares (ALS)\n"
            info += "✓ Framework: PySpark MLlib\n"
            info += "✓ Método de similitud: Locality-Sensitive Hashing (LSH)\n\n"
            info += "Descripción:\n"
            info += "-" * 60 + "\n"
            info += "Este modelo utiliza factorización matricial (ALS) para generar\n"
            info += "vectores latentes de usuarios basándose en sus gustos musicales.\n"
            info += "La similitud entre usuarios se calcula usando LSH (Bucketed Random\n"
            info += "Projection) sobre estos vectores latentes.\n"
            
            self.model_info_text.insert(tk.END, info)
            
        self.model_info_text.config(state='disabled')
        
    def generar_recomendaciones(self):
        """Genera recomendaciones de usuarios similares"""
        if not self.sistema.modelo_cargado:
            messagebox.showerror(
                "Error",
                "Por favor carga el modelo primero"
            )
            return
            
        if not self.sistema.usuario_actual:
            messagebox.showerror(
                "Error",
                "Por favor selecciona un usuario primero"
            )
            return
            
        try:
            top_k = int(self.num_recs.get())
        except ValueError:
            messagebox.showerror(
                "Error",
                "Número de recomendaciones inválido"
            )
            return
            
        # Mostrar mensaje de carga
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Generando recomendaciones...\n")
        self.results_text.config(state='disabled')
        
        # Ejecutar en thread separado
        def _generar():
            resultados, error = self.sistema.recomendar_usuarios_similares(
                self.sistema.usuario_actual,
                top_k
            )
            self.root.after(0, lambda: self._mostrar_recomendaciones(resultados, error))
            
        Thread(target=_generar, daemon=True).start()
        
    def _mostrar_recomendaciones(self, resultados, error):
        """Muestra las recomendaciones en la interfaz"""
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        
        if error:
            self.results_text.insert(tk.END, f"❌ Error: {error}\n")
        elif not resultados:
            self.results_text.insert(tk.END, "No se encontraron usuarios similares\n")
        else:
            # Encabezado
            header = f"Usuarios similares al Usuario {self.sistema.usuario_actual}\n"
            header += "="*70 + "\n\n"
            self.results_text.insert(tk.END, header, 'header')
            
            # Resultados
            for i, row in enumerate(resultados, 1):
                if hasattr(row, 'user_id'):
                    # Con información de username
                    user_id = row.user_id
                    distance = row.distCol
                    username = row.Username if hasattr(row, 'Username') else "N/A"
                    
                    self.results_text.insert(tk.END, f"{i:2d}. ", 'header')
                    self.results_text.insert(tk.END, f"Usuario {user_id}", 'user_id')
                    self.results_text.insert(tk.END, f" | {username}", 'username')
                    self.results_text.insert(tk.END, f" | Distancia: {distance:.4f}", 'distance')
                    self.results_text.insert(tk.END, "\n")
                else:
                    # Sin información de username
                    user_id = row.id
                    distance = row.distCol
                    
                    self.results_text.insert(tk.END, f"{i:2d}. ", 'header')
                    self.results_text.insert(tk.END, f"Usuario {user_id}", 'user_id')
                    self.results_text.insert(tk.END, f" | Distancia: {distance:.4f}", 'distance')
                    self.results_text.insert(tk.END, "\n")
            
            self.results_text.insert(tk.END, "\n" + "="*70 + "\n")
            self.results_text.insert(tk.END, f"Total: {len(resultados)} usuarios encontrados")
            
        self.results_text.config(state='disabled')
    
    def cargar_historial(self):
        """Carga el historial de canciones del usuario"""
        if not self.sistema.modelo_cargado or not self.sistema.spark:
            messagebox.showerror(
                "Error",
                "Por favor carga el modelo primero (pestaña Modelo)"
            )
            return
        
        # Obtener el ID del usuario del campo de historial
        try:
            user_id = int(self.historial_user_entry.get())
        except ValueError:
            messagebox.showerror(
                "Error",
                "Por favor ingresa un ID de usuario válido"
            )
            return
            
        try:
            limit = int(self.num_songs.get())
        except ValueError:
            limit = 50
            
        # Mostrar mensaje de carga
        self.historial_text.config(state='normal')
        self.historial_text.delete(1.0, tk.END)
        self.historial_text.insert(tk.END, f"Cargando historial del usuario {user_id}...\n")
        self.historial_text.config(state='disabled')
        
        # Ejecutar en thread separado
        def _cargar():
            resultados, error = self.sistema.obtener_canciones_usuario(user_id, limit)
            # También cargar estadísticas
            stats, _ = self.sistema.obtener_estadisticas_usuario(user_id)
            self.root.after(0, lambda: self._mostrar_historial(resultados, error, stats, user_id))
            
        Thread(target=_cargar, daemon=True).start()

    def _mostrar_historial(self, resultados, error, stats=None, user_id=None):
        """Muestra el historial de canciones en la interfaz"""
        self.historial_text.config(state='normal')
        self.historial_text.delete(1.0, tk.END)
        
        # Actualizar estadísticas rápidas
        if stats:
            self.stats_total_songs.config(text=f"Total canciones: {stats.total_canciones:,}")
            self.stats_total_plays.config(text=f"Total reproducciones: {int(stats.total_reproducciones):,}")
            self.stats_avg_plays.config(text=f"Promedio por canción: {stats.promedio_reproducciones:.1f}")
        else:
            self.stats_total_songs.config(text="Total canciones: --")
            self.stats_total_plays.config(text="Total reproducciones: --")
            self.stats_avg_plays.config(text="Promedio por canción: --")
        
        if error:
            self.historial_text.insert(tk.END, f"❌ Error: {error}\n")
        elif not resultados:
            self.historial_text.insert(tk.END, "No se encontraron canciones para este usuario\n")
        else:
            # Encabezado
            header = f"🎵 HISTORIAL DEL USUARIO {user_id}\n"
            header += "="*120 + "\n\n"
            self.historial_text.insert(tk.END, header, 'header')
            
            # Calcular total de reproducciones mostradas
            total_plays = sum(row.play_count for row in resultados)
            
            self.historial_text.insert(tk.END, f"Reproducciones mostradas: {total_plays:,}\n", 'play_count')
            self.historial_text.insert(tk.END, f"Canciones mostradas: {len(resultados)}\n\n", 'play_count')
            
            # Encabezado de tabla
            self.historial_text.insert(tk.END, "-"*120 + "\n", 'separator')
            header_row = f"{'#':<4} {'SONG ID':<22} {'CANCIÓN':<35} {'ARTISTA':<35} {'PLAYS':<10}\n"
            self.historial_text.insert(tk.END, header_row, 'header')
            self.historial_text.insert(tk.END, "-"*120 + "\n", 'separator')
            
            # Resultados
            for i, row in enumerate(resultados, 1):
                song_id = str(row.song_id) if hasattr(row, 'song_id') else "N/A"
                play_count = row.play_count if hasattr(row, 'play_count') else 0
                
                # Obtener nombre de canción y artista si existen
                song_name = str(row.title)[:33] if hasattr(row, 'title') and row.title else "Desconocida"
                artist = str(row.artist_name)[:33] if hasattr(row, 'artist_name') and row.artist_name else "Desconocido"
                
                # Formatear fila
                self.historial_text.insert(tk.END, f"{i:<4} ")
                self.historial_text.insert(tk.END, f"{song_id:<22}", 'song_id')
                self.historial_text.insert(tk.END, f"{song_name:<35}", 'song_name')
                self.historial_text.insert(tk.END, f"{artist:<35}", 'artist')
                self.historial_text.insert(tk.END, f"{play_count:>8,}\n", 'play_count')
            
            self.historial_text.insert(tk.END, "-"*120 + "\n", 'separator')
            self.historial_text.insert(tk.END, f"\n✓ Mostrando top {len(resultados)} canciones más escuchadas del usuario {user_id}", 'header')
            
        self.historial_text.config(state='disabled')
    
    def mostrar_estadisticas(self):
        """Muestra estadísticas detalladas del usuario"""
        if not self.sistema.modelo_cargado or not self.sistema.spark:
            messagebox.showerror(
                "Error",
                "Por favor carga el modelo primero (pestaña Modelo)"
            )
            return
        
        # Obtener el ID del usuario del campo de historial
        try:
            user_id = int(self.historial_user_entry.get())
        except ValueError:
            messagebox.showerror(
                "Error",
                "Por favor ingresa un ID de usuario válido"
            )
            return
        
        # Mostrar mensaje de carga
        self.historial_text.config(state='normal')
        self.historial_text.delete(1.0, tk.END)
        self.historial_text.insert(tk.END, f"Calculando estadísticas del usuario {user_id}...\n")
        self.historial_text.config(state='disabled')
        
        def _calcular():
            stats, error = self.sistema.obtener_estadisticas_usuario(user_id)
            self.root.after(0, lambda: self._mostrar_estadisticas_detalladas(stats, error, user_id))
            
        Thread(target=_calcular, daemon=True).start()
    
    def _mostrar_estadisticas_detalladas(self, stats, error, user_id):
        """Muestra las estadísticas detalladas"""
        self.historial_text.config(state='normal')
        self.historial_text.delete(1.0, tk.END)
        
        if error:
            self.historial_text.insert(tk.END, f"❌ Error: {error}\n")
        elif not stats:
            self.historial_text.insert(tk.END, "No se encontraron estadísticas para este usuario\n")
        else:
            # Actualizar labels de estadísticas rápidas
            self.stats_total_songs.config(text=f"Total canciones: {stats.total_canciones:,}")
            self.stats_total_plays.config(text=f"Total reproducciones: {int(stats.total_reproducciones):,}")
            self.stats_avg_plays.config(text=f"Promedio por canción: {stats.promedio_reproducciones:.1f}")
            
            # Mostrar estadísticas detalladas
            header = f"📊 ESTADÍSTICAS DEL USUARIO {user_id}\n"
            header += "="*70 + "\n\n"
            self.historial_text.insert(tk.END, header, 'header')
            
            self.historial_text.insert(tk.END, "RESUMEN GENERAL\n", 'stats')
            self.historial_text.insert(tk.END, "-"*70 + "\n\n", 'separator')
            
            self.historial_text.insert(tk.END, "  🎵 Total de canciones únicas: ", 'play_count')
            self.historial_text.insert(tk.END, f"{stats.total_canciones:,}\n", 'song_id')
            
            self.historial_text.insert(tk.END, "  🔁 Total de reproducciones: ", 'play_count')
            self.historial_text.insert(tk.END, f"{int(stats.total_reproducciones):,}\n", 'song_id')
            
            self.historial_text.insert(tk.END, "  📈 Promedio de reproducciones por canción: ", 'play_count')
            self.historial_text.insert(tk.END, f"{stats.promedio_reproducciones:.2f}\n\n", 'song_id')
            
            self.historial_text.insert(tk.END, "="*70 + "\n", 'separator')
            self.historial_text.insert(tk.END, "\n💡 Usa 'Cargar Historial' para ver las canciones específicas", 'header')
            
        self.historial_text.config(state='disabled')


def main():
    """Función principal"""
    root = tk.Tk()
    app = InterfazSoundTrack(root)
    root.mainloop()


if __name__ == "__main__":
    main()