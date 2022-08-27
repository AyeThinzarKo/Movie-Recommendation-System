import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

with st.sidebar:
  selected=option_menu(
    menu_title='MovRec',
    options=['Home','Best Rated','Trending'],
    icons=['house','asterisk','bell'],#bell-fill
    menu_icon='cast',
    default_index=0,
    )

#selected=option_menu(menu_title="Main Menu", options=['Home','Best Rated','Trending'],icons=['house-fill','asterisk','bell-fill'],menu_icon='cast',default_index=0,orientation='horizontal')

#----importing dataset----

df=pd.read_csv('movies.csv')
#st.write(df.columns)
#df

#----------Home------------------

if selected=='Home':
  st.title('Movie Recommendation System')
  st.markdown("***")
  st.subheader('Discover new movies')

  movie_name=st.text_input('Enter a movie title and get best recommended movies based on its genres')
  #st.write(movie_name)

  features=['keywords','cast','genres','director']
  #st.write(features)

  def combine_features(row):
    return row['keywords']+' '+row['cast']+' '+row['genres']+' '+row['director']

  for feature in features:
    df[feature]=df[feature].fillna('')
  df['combined_features']=df.apply(combine_features,axis=1)
  #df.iloc[0].combined_features

  #create count matrix from this new combined column

  vectorizer=CountVectorizer()
  count_matrix=vectorizer.fit_transform(df['combined_features'])

  cosine_sim=cosine_similarity(count_matrix)

  def get_title_from_index(index):
    return df[df.index==index]['title'].values[0]

  def get_index_from_title(title):
    return df[df.title==title]['index'].values[0] 

  #-----Button----Recommend---


  if st.button('Recommend'):
    #st.write(df['title'].values)
    if movie_name=='':
      st.info('Please fill movie name')

    elif movie_name in df['title'].values:
      movie_user_likes=movie_name

      movie_index=get_index_from_title(movie_user_likes)
      #st.write(movie_index)
      similar_movies=list(enumerate(cosine_sim[movie_index]))

      sorted_similar_movies=sorted(similar_movies,key=lambda x:x[1],reverse=True)

      #print titles of first 5 movies
      st.write('Recommended movies for you: \n')
      i=0
      col1,col2=st.columns([1,2])
        
      with col1:
        st.write('Movie Name')
      with col2:
        st.write('Genres:')
        
      for movie in sorted_similar_movies:
       
        with col1:
          movie_result=get_title_from_index(movie[0])
          st.write(movie_result)
        with col2:
          st.write(df[df.title==movie_result]['genres'].values[0])
       
        i=i+1
        if i>5:
          break  
    else:
      st.info('Sorry, this movie is not recommended for you')
      

#-----------------------------------------------


#-----------Best Rated------------------

if selected=='Best Rated':
  st.header('Best - rated Movies')
  st.write("\n")
  #df

  C= df['vote_average'].mean()
  #st.write(C)
  m=df['vote_count'].quantile(0.9)
  #st.write(m)
  q_movies=df.copy().loc[df['vote_count']>=m]
  #q_movies.shape

  # Calculation of weighted rating based on the IMDB formula
  def weighted_rating(x, m=m, C=C):
      v = x['vote_count']
      R = x['vote_average']
      return (v/(v+m) * R) + (m/(m+v) * C)

  # Add new feature 'score' and calculate its value with `weighted_rating()`
  q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

  #Sort movies based on score
  q_movies = q_movies.sort_values('score', ascending=False)
  #st.write(type(q_movies))

  #Print the top rated movies
  #q_movies[['title', 'vote_count', 'vote_average', 'score']] 


  col1,col2,col3,col4 = st.columns(4)

  with col1:
    image = Image.open('images\\img1.jpg')
    st.image(image, caption='Shawshank Redemption')

  with col2:
    image = Image.open('images\\img2.jpeg')
    st.image(image, caption='Fight Club')

  with col3:
    image = Image.open('images\\img3.jpg')
    st.image(image, caption='The Dark Knight')

  with col4:
    image = Image.open('images\\img4.png')
    st.image(image, caption='Pulp Fiction')


  with col1:
    image = Image.open('images\\img5.jpg')
    st.image(image, caption='Inception')

  with col2:
    image = Image.open('images\\img6.jpg')
    st.image(image, caption='The GodFather')

  with col3:
    image = Image.open('images\\img7.jpg')
    st.image(image, caption='Interstellar')

  with col4:
    image = Image.open('images\\img8.jpg')
    st.image(image, caption='Forest Gump')


  with col1:
    image = Image.open('images\\img9.jpg')
    st.image(image, caption='The Load of the Rings: The Return of the King')

  with col2:
    image = Image.open('images\\img10.jpg')
    st.image(image, caption='The Empire Strikes Back')

  with col3:
    image = Image.open('images\\img11.jpg')
    st.image(image, caption='The Lord of the Rings: The Fellowship of the Ring')

  with col4:
    image = Image.open('images\\img12.jpg')
    st.image(image, caption='Star Wars')
  
  #-----------------------------------------------
  
  #--------Tredending-------------------
 

if selected=='Trending':
  st.header('Trending Popular Movies')
  st.write("\n")
  p_movies = df.sort_values('popularity', ascending=False)
  #st.write(p_movies[['title']])

  col1,col2,col3,col4 = st.columns(4)

  with col1:
    image = Image.open('images2\\img1.jpg')
    st.image(image, caption='Minions')

  with col2:
    image = Image.open('images2\\img2.jpg')
    st.image(image, caption='Interstellar')

  with col3:
    image = Image.open('images2\\img3.jpg')
    st.image(image, caption='Deadpool')

  with col4:
    image = Image.open('images2\\img4.jpg')
    st.image(image, caption='Guardians of the Galaxy')


  with col1:
    image = Image.open('images2\\img5.jpg')
    st.image(image, caption='Mad Max: Fury Road')

  with col2:
    image = Image.open('images2\\img6.jpg')
    st.image(image, caption='Jurassic World')

  with col3:
    image = Image.open('images2\\img7.jpg')
    st.image(image, caption='Pirates of the Caribbean')

  with col4:
    image = Image.open('images2\\img8.jpg')
    st.image(image, caption='Dawn of the Planet of Apes')


  with col1:
    image = Image.open('images2\\img9.jpg')
    st.image(image, caption='The Hunger Games: Mockingjay - Part 1')

  with col2:
    image = Image.open('images2\\img10.jpg')
    st.image(image, caption='Big Hero 6')

  with col3:
    image = Image.open('images2\\img11.jpg')
    st.image(image, caption='Terminator Genisys')

  with col4:
    image = Image.open('images2\\img12.jpg')
    st.image(image, caption='Captain America: Civil War')
  
  #-----------------------------------------------


