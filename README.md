# multiple-regression-sales-volume
Predicting sales volume for different product types using multiple regression, and analyzing the impact customer reviews have on sales.

Given historical sales data, the goal is to make sales volume predictions for a list of new product types. The client is most interested in sales volume of four product types - PCs, Laptops, Netbooks, and Smartphones - but we will be analyzing other types as well. 

The given data includes the following information:

- "ProductType" - which names the product type for each product
- "ProductNum" - an integer that indicates the product ID number
- "x5StarReviews", "x4StarReviews", etc - an integer indicating the number of 5 star reviews, 4 star reviews, 3 star, 2 star, and 1 star reviews
- "PositiveServiceReview" - an integer indicating the number of positive service reviews a product has recieved
- "NegativeServiceReview" - and integer indicating the number of negative service reviews a product has recieved
- "RecommendProduct" - a number on a scale from 0 to 1 indicating whether customers would recommend the product
- "BestSellersRank" - a number indicating what a product's best seller rank is -- not all products are included in the best seller category, so there are NA values
- "ShippingWeight" - indicating a product's shipping weight
- "ProductDepth" - indicating a product's measured depth
- "ProductWidth" - indicating a product's measured width
- "ProductHeight" - indicating a product's measured height
- "ProfitMargin" - indicating the profit margin for that product
- "Volume" - indicating the sales volume for a given product

Three different regression models were tested - SVM, Random Forest, and Gradient boosting. Unfortunately, the models overfit the data due to the small sample size and outliers in the data. Detailed results can be viewed in the "C3T3 Report.docx" file, as well as the "newproductspreds.csv" file.
