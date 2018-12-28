# BGG_Data_Analysis
a python3 code that has the following functions:
1. scrap game ids from bgg
2. mine the games' data through the xml API
3. make some analysis to aid finding the most suitable game for a specific number of players
4. mine user ratings for each game
5. make recommendations based on similar users

# dependencies
`pip3 install --user requests, bs4, pandas, adjustText, mplcursors`

# instructions
to use the analysis part, run the script "analyse_bgg_data".
for first run:
 - set MINE_INFO_FROM_BGG to 1
 - uncomment the line `path_gameIDs = ''`
 
after you've done it once, the data is saved in your repository folder, so you can set back MINE_INFO_FROM_BGG to 0 and recomment the line.

# References
all of the user ratings mining and recommendation algorithm is heavily based on the work from:
http://sdsawtelle.github.io/blog/output/boardgamegeek-data-scraping.html
  