# Video Game Sales Analysis
## by Sean Zhong


## Dataset

> My dataset contains video game sales data scraped from site 'VGChartz' on 10/26/2016, and downloaded from Kaggle. It only contains video sales data (such as Name, Platform, Year, Genre, Publisher, and Regional Sales) from 1980 to 2016. 

> Before starting the data exploration, I first downloaded the file programmatically from Kaggle, before importing into a Jupyter Notebook. I then ran various summary statistics to gain an understanding of the data, as well as to see if there were any cleaning steps I would have to conduct. From this, I identified 4 data quality issues, which I then proceeded to clean. Once cleaned, I started my data exploration and analysis.


## Summary of Findings

> I've gathered below a summary of my findings:

### Platform
- The most popular platforms in terms of game releases were the DS and PS2, while the most popular platform in terms of game sales was the PS2 by a wide margin, with the DS dropping down to 5th place.

### Genre
- Over the years, the **'Action'** and **'Sports'** genres have been the most popular in terms of both the number of releases and global sales.
- The **'Puzzle'** and **'Strategy'** genres have remained at or near the bottom in terms of global sales and number of releases.
- For every genre, except **'Role-Playing'**, NA sales top the other regions.
- The NA and EU regions are consistently the top two regions by sales volume for most genres.
- The **'Role-Playing'** genre is clearly strongest in Japan, while also being that regions most popular genre.
- The most popular genre in the NA and EU regions is the **'Action'** genre.
- The **'Action'** genre has remained dominant in terms of global sales for the longest, lasting until the end of the dataset period coverage.
- In terms of global sales, the **'Sports'** genre has started to fall off in recent years, being overtaken by the **'Shooter'** genre as the second most popular in terms of sales.

### Publisher
- In terms of games released, **Electronic Arts (EA)** is consistently the biggest publisher each year; however, in terms of global sales, **Nintendo** is the biggest, though not consistently when broken down by year.
- Though **Nintendo** is technically the biggest publisher in terms of global sales in aggregate, its global sales were greatly bolstered in 2006 by Wii Sports, a free game that came included with the Wii console platform. In 2006, Wii Sports accounted for almost half of **Nintendo's** global sales for that year.
- In recent years, **EA** has been extremely competitive with **Nintendo** in terms of global sales, actually beating **Nintendo** in the last two years of sales data.

### Sales
- All sales regions followed the same general trendline, peaking around 2008. However, Japan sales remained relatively flat over the years, when compared to the other regions.
- North American sales were almost an exact mirror of the Global Sales trendline, while also having the most sales among all the regions. This seems to indicate that North American sales are a major driver of overall Global Sales.
- In the early years (1980 - 1993), Global Sales and the number of games released were closely inline with each other; however, they started to pull apart in 1995 as they both started to increase.
- The rate of increase for the number of games being released greatly outpaced that of Global Sales, indicating that sales could not keep up. Thus, while total sales may have been increasing up till 2008, average sales per game were actually decreasing.

> From the above findings, I will be concentrating on only the Genre conclusions in my explanatory presentation.

## Key Insights for Presentation

> Below is the main thread of analysis for my explanatory presentation:

### Genre
- For nearly every genre, North American sales top the other regions.
- However, the sole exception is the Role-Playing genre, which is clearly strongest in Japan. Coincidentally, it is also that regions most popular genre.
- Meanwhile, the most popular genre in North America is the Action genre.
- Historically, the Action and Sports genres have been the best global sellers when summed over all the years.
- Additionally, breaking global sales down by year, the Action genre has remained the best selling genre for the longest time out of all genres, remaining the best seller through the end of the dataset period coverage.
- On the other hand, the historically second best selling genre (Sports) has started to drop off in terms of sales in recent years.
- In the last few years, it has been overtaken by the Shooter genre as the second most popular in terms of sales.

