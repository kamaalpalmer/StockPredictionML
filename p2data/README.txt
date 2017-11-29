Files:

1. Original Data set with all of the attributes, missing values, and 
extraneous characters.
2. Transformed and cleaned data
3. quartiles.py to generate column headers and new quartile column for class
attribute.
4. Readme.txt

Summary:

I took the original dataset and removed any extra data or characters. First
I removed the column with the date and the stock ticker symbol which is not
needed for machine learning. Second I removed the rows with missing
values. These rows included the percent change over last weeks price and the
previous weeks volume which I couldn't input dummy values for and these
numbers may make a significant impact so it was important to remove the entire
row. After that I moved the class attribute to the last column and wrote a
quartiles program to divide the percent change in next weeks price, the class 
attribute, into low, medium, and high quartiles, and create a new column for
this. I also removed the dollar signs with a find/replace all.

