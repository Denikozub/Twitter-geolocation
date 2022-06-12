# Twitter users geolocation based on tweet texts

## Data
Processed [dataset](https://archive.org/details/twitter_cikm_2010) contains 620k tweets and corresponding coordinates.  
Processing includes geocoding US cities, which is done using Nominatim, and country location check.  
Train - test - val split: 80% - 10% - 10%.  
The task is to predict coorditates (lat - lon) based on tweet texts.
