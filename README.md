Câu 2:
df_akas.filter(F.col("language").isNotNull()).groupby("language").count()\
       .orderBy("count", ascending=False)\
       .show(3)

Câu 3:
df_genre_pick = df_basics.filter((F.col("titleType")=="movie") & (F.col("startYear")>=2010) & (F.col("startYear")<=2015) )\
         .withColumn("arr_genres", F.split("genres", ","))\
         .select("tconst", F.explode("arr_genres").alias("genre"))
         
df_film_pick = df_akas.filter(F.col("region")=="US").withColumn('tconst', F.col('titleId'))


df_ratings.join(df_genre_pick, on='tconst').join(df_film_pick, on="tconst")\
    .groupby("genre").agg(F.avg("averageRating").alias("rating"))\
    .orderBy("rating", ascending=False)\
    .show(5).

Câu 4:

"""Nhận xét: nếu các đạo diễn có xu hướng lựa chọn diễn viên thân thuộc là đúng, sẽ có rất nhiều bộ phim có sự hợp tác của họ
=> Vẽ phân phối số lần hợp tác sẽ đánh giá được giả thiết
"""
# Lấy tập data theo yêu cầu đề bài
movie = df_basics.filter((F.col("titleType")=="movie") &  F.col("startYear").between(2010, 2015) )
us = df_akas.filter(F.col("region")=="US").selectExpr('titleId AS tconst')

principals = df_principals.filter( F.col('category').isin('actor', 'actress', 'director') )\
                .join(movie, on='tconst').join(us, on='tconst')

# Đếm số lần hợp tác
actor = principals.filter(F.col('category').isin('actor', 'actress')).withColumn('actor', F.col('nconst'))
director = principals.filter(F.col('category')=='director').withColumn('director', F.col('nconst'))

colaboration = actor.join(director, on='tconst').groupby('actor', 'director').agg(F.count('tconst').alias('movie_num'))\
                .withColumn('is_1time', F.when(F.col('movie_num')==1, 1).otherwise(0))

his = colaboration.groupby('is_1time').count()
print('Số bộ phim: ', actor.join(director, on='tconst').select(F.countDistinct('tconst')).collect())
print('Số lần hợp tác:')
his.show()

Trung bình 1 bộ phim có 100 diễn viên thì sẽ có khoảng 11 diễn viên cũ (từng hợp tác với đạo diễn)
=> Các diễn viên và đạo diễn có xu hướng hợp tác với nhau nhiều lần

Câu 5:
film = df_basics.filter(F.col("titleType")=="movie").select("tconst")
director = df_person.filter(F.col("deathYear").isNull()).select('nconst', 'primaryProfession')
ratings = df_ratings.filter(F.col("numVotes")>200)

principals = df_principals.filter( F.col('category')=='director' ).select('nconst', 'tconst')

matched = ratings.join(film, on='tconst')\
                .join(principals, on='tconst')\
                .join(director, on='nconst')\
                .withColumn('is_bad', F.when(F.col('averageRating') <=4, 1).otherwise(0))\
                .withColumn('is_good', F.when(F.col('averageRating') >7.5, 1).otherwise(0))\
                .withColumn('is_writer', F.when(F.col('primaryProfession').contains('writer'), 1).otherwise(0) ).persist()

smr_by_director = matched.groupby("nconst", "is_writer").agg(
    F.sum("is_bad").alias('bad'),
    F.sum("is_good").alias('good'),
    F.count("tconst").alias('total'),
    (F.sum("is_bad")/F.count("tconst")).alias("bad_ratio")
)
smr = smr_by_director.filter(
    (F.col('bad')<=10 ) &
    (F.col('good')>5) &
    (F.col('bad_ratio')<=0.2)
).selectExpr("COUNT(1) AS dir_num", 
             "SUM(is_writer) AS writer", 
             "SUM(is_writer)/COUNT(1) AS pct" 
             ).collect()
writer_num = smr[0].writer
dir_num = smr[0].dir_num


"""Với số đạo diễn đề cử là dir_num, số đạo diễn có kiêm biên kịch là writer_num
Số cách chọn 3 đạo diễn từ writer_num là x = C(writer_num, 3)
Số cách chọn 3 đạo diễn từ dir_num là y = C(dir_num, 3)
Xác suất để 3 đạo diễn là đồng thời là biên kịch là: p = x/y
"""

import math

def comb(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

p = comb(writer_num, 3)/ comb(dir_num,3)

print(p)








from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor

def array_to_onehot(df, inputCol, sep=',', id_col = 'tconst'):
  df = df.withColumn('arr_'+inputCol, F.split(inputCol, sep))\
          .select(id_col, F.explode('arr_'+inputCol).alias("element_"+inputCol))\
          .withColumn('value', F.lit(1))
  df_vec = df.groupby(id_col).pivot("element_"+inputCol).count()
  return df_vec

def cast_col(df, col, dtype='int'):
  df = df.withColumnRenamed(col, col+'tmp')
  df = df.withColumn(col, F.col(col+'tmp').cast(dtype))
  return df.drop(col+'tmp')
def cast_cols(df, cols, dtype='int'):
  for col in cols:
    df = cast_col(df, col)
  return df



# YOUR CODE START 
akas = df_akas.groupby('titleId').agg(F.count('titleId').alias('region_num'))

full = df_basics.filter((F.col("titleType")=="movie") & F.col("startYear").between(2000,2021))\
        .join(akas, on = [df_basics.tconst == df_akas.titleId], how='left')\
        .join(df_ratings, on='tconst', how='left')\
        .withColumn('dataset', F.when(F.col("startYear").between(2000,2015), 'train').otherwise('test'))

num_cols = ['isAdult','startYear',  'runtimeMinutes', 'region_num', 'averageRating']
full = cast_cols(full, num_cols, dtype='double')

# Optional: adding genre information to features
df_genre = array_to_onehot(full, inputCol='genres')
full = full.join(df_genre, on='tconst', how='left').fillna(0)
cols_genres = df_genre.drop('tconst').columns

pipe = Pipeline(stages=[
        pyspark.ml.feature.VectorAssembler(inputCols= num_cols+cols_genres, outputCol='fts'),
        RandomForestRegressor( featuresCol = 'fts', labelCol = 'averageRating', predictionCol='pred',  
                              numTrees=10 )
])

model = pipe.fit(full.filter(F.col('dataset')=='train'))

df_pred = model.transform(full.filter(F.col('dataset')=='test')).select('tconst', 'pred')


# / END YOUR CODE








# YOUR CODE START 
# DO NOT CHANGE THIS CODE OR THE TESTS MAY NOT WORK
vocab_size = 1000
embedding_dim = 16
max_length = 120
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000

sentences = []
labels = []

# train 
train_sentences = train['originalTitle']
valid_sentences = test['originalTitle']
# label 
train_labels = train['isAdult']
valid_labels = test['isAdult']

token = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token= oov_tok)
token.fit_on_texts(train_sentences)
word_index = token.word_index

train_sequences = token.texts_to_sequences(train_sentences)
valid_sequences = token.texts_to_sequences(valid_sentences)

train_padded = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, truncating=trunc_type, padding=padding_type, maxlen=max_length)
valid_padded = tf.keras.preprocessing.sequence.pad_sequences(valid_sequences, truncating=trunc_type, padding=padding_type, maxlen=max_length)

train_labels = np.asarray(train_labels).astype('int')
valid_labels = np.asarray(valid_labels).astype('int')

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_padded, train_labels, epochs=3, batch_size=1024)

prd = model.predict(valid_padded)
df_pred = pd.DataFrame({'tconst':test.tconst, 'pred': prd.reshape(prd.shape[0])})

# / END YOUR CODE
