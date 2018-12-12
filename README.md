# BGG_Data_Analysis
this is a short python3 code that:
1. scraps game ids from bgg
2. fetch their data through the xml API
3. make simple analysis to aid finding the most suitable game for a specific number of players

# dependencies
`pip3 install --user requests, bs4, pandas, adjustText, mplcursors`

# instructions
to use the code, run the script "analyse_bgg_data".
for first run:
 - set MINE_INFO_FROM_BGG to 1
 - uncomment the line `path_gameIDs = ''`
 
after you've done it once, the data is saved in your repository folder, so you can set back MINE_INFO_FROM_BGG to 0 and recomment the line.
