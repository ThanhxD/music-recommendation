import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="",
  database="recommend_music"
)

mycursor = mydb.cursor()

def listSongs():
    mycursor.execute("SELECT * FROM song JOIN artist ON song.artist = artist.id")
    results = mycursor.fetchall()
    return results
