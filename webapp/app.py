from flask import Flask, session, render_template, request
from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)
app.secret_key = (
    "kjhfdskjhfsdghk;odgkjsdkjsdkgsdgjdsghdjgdghdfghdfgdfghfggdf;uaghdfgagf"
)
app.config["SESSION_TYPE"] = "filessystem"

basepath = os.path.abspath(".")

# Load bayes model
with open(basepath + '/static/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load kmeans model
with open(basepath + '/static/kmeans_model.pkl', 'rb') as model_file2:
    model2 = pickle.load(model_file2)

# Load CSV files into dataframes
df = pd.read_csv(basepath + '/static/df.csv')
X_scaled_df = pd.read_csv(basepath + '/static/X_scaled_df.csv')

# Create dictionary structure
artist_album_song_dict = {}
for artist, album, song in zip(df['artists'], df['album_name'], df['track_name']):
    if artist not in artist_album_song_dict:
        artist_album_song_dict[artist] = {}
    if album not in artist_album_song_dict[artist]:
        artist_album_song_dict[artist][album] = []
    artist_album_song_dict[artist][album].append(song)

# Route for main page
@app.route('/', methods=['POST', 'GET'])
def index():

    # If post
    if request.method == 'POST':
        text = request.form['text']
        predicted_label = request.form['prediction']
        predicted_label = model.predict([text])
        if(predicted_label == 1):
            predicted_label = 'Prediction is SPAM'
        elif(predicted_label == 0):
            predicted_label = 'Prediction is not SPAM'
        print("POST", predicted_label)
        return render_template('index.html', prediction=predicted_label, text=text)

    else:
        print("GET")
        return render_template('index.html', prediction='', input_text='')

# Route for song recommender
@app.route('/songs', methods=['POST', 'GET'])
def songs():
    return render_template(
        "songs.html",
        song_data=sorted(artist_album_song_dict.keys()),
    )


@app.post("/artists")
def get_album():
    the_artist = request.form["artist"]
    print("The artist", the_artist)
    session["the_artist"] = the_artist
    album_keys= artist_album_song_dict[the_artist].keys()
    albums = list(album_keys)
    # Add - to index 0, to be first value shown
    albums.insert(0, "-")
    print("Albums", albums)
    return render_template(
        "select.html",
        data=sorted(albums),
    )

@app.post("/album")
def get_song():
    the_album = request.form["album"]
    #print("The album", the_album)
    session["the_album"] = the_album
    the_artist = session["the_artist"]
    songs= artist_album_song_dict[the_artist][the_album]
    # Add - to index 0, to be first value shown
    songs.insert(0, "-")
    #print("Songs", songs)
    return render_template(
        "select.html",
        data=sorted(songs),
    )

@app.post("/song")
def get_songs():
    the_song = request.form["song"]
    the_artist = session["the_artist"]
    the_album = session["the_album"]
    song_recommend = recommender(df, the_artist, the_album, the_song,model2,X_scaled_df)
    tracks = get_song_info(df, song_recommend)
    tracks_str = ''.join(tracks)
    #print("Tracks: ", tracks)
    # Add - to index 0, to be first value shown
    return render_template(
        "textarea.html",
        songs=tracks_str,
    )


def recommender(df, artist_name, album_name, track_name, kmeans, X_scaled_df):
    # Find the song in the dataset
    song_index = df[(df['artists'] == artist_name) & (df['album_name'] == album_name) & (df['track_name'] == track_name)].index
    if len(song_index) == 0:
        print("Song not found.")
        return None

    # Get song cluster index
    song_cluster_index = X_scaled_df.loc[song_index, 'cluster'].values[0]
    
    # Filter dataframe to include only songs from the same cluster
    cluster_df = X_scaled_df[X_scaled_df['cluster'] == song_cluster_index]

    # Get song cluster values
    song_cluster = cluster_df.drop(columns=['cluster']).values

    # Get centroid of cluster
    cluster_centroid = kmeans.cluster_centers_[song_cluster_index]

    # Calculate pairwise distances for songs in cluster
    distances = pairwise_distances( song_cluster, [cluster_centroid])

    # Find the indices of the closest songs
    closest_indices = np.argsort(distances.flatten())[:10]

    # Get track IDs of closest songs
    closest_track_ids = df.iloc[closest_indices]['track_id'].tolist()
    
    return closest_track_ids   

def get_song_info(df, track_ids):
    tracks = []
    rec_id = 1
    for id in track_ids:
        artist, track_name, album_name = df.loc[df['track_id'] == id, ['artists', 'track_name', 'album_name']].values[0]
        track_info = f"Recommendation #{rec_id}\nArtist Name: {artist}\nTrack Name: {track_name}\nAlbum Name: {album_name}\n\n"
        rec_id += 1
        tracks.append(track_info)
    return tracks

if __name__ == '__main__':
    app.run(debug=True)