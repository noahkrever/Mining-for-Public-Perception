README.txt

In this project, we queried hundreds of recent tweets and the top 25 articles returned by Google news that contained the names of the top 100 companies in the Forbes Fortune 1000 dataset. We then perform sentiment analysis on these tweets and articles to create a selection of sentiment-related features, like polarity and subjectivity. Next, we apply correlation and regression analysis to unearth relationships in the data between these features, and characteristics of the companies, like success markers, and different industries. Finally, we take a look at the relationships we were able to generate, and attempt to draw some conclusions about them. We were able to pinpoint a few interesting insights, but the main value of this project is the validation of the process of text scraping and sentiment analysis, followed by joining on a more traditional dataset, and application of data mining algorithms, that we used.

File Information:

Just a note, these files are very raw, and simply in the last form they were used in. They changed a lot over the course of this project, but all of the main functions and processes should be visible in them.

backend.py: These are the python scripts we used to initially query the tweets and google news articles. It also contains the functions we used to iterate over the fortune 1000 dataset and compile information for each company into one overall csv file.

Project3.ipynb: This is the iPython notebook we used to join the datasets, run correlation, and regression, as well as generate the plots and WordClouds you see in the written report. The main "data mining" processes are done here.

corrplot.Rmd: This is the rmd file we used to produce the corrplot in the report.

Project 3 Written Report.pdf: This is the written report of the project.