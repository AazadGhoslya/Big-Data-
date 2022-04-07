# Databricks notebook source
# MAGIC %md
# MAGIC ### Big Data Tools
# MAGIC ##### MBD 2021 - 2022
# MAGIC ##### Team Members: THAYANIDHI Kamalakannan, GHOSLYA Aazad , ROMAN INFANTE Pedro

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Loading Data

# COMMAND ----------

# Loading data
filePath = "/FileStore/tables/parsed_data/parsed_data"
covid_df = spark.read.json(filePath+"/parsed_covid.json")
checkin_df = spark.read.json(filePath+"/parsed_checkin.json")
tips_df = spark.read.json(filePath+"/parsed_tip.json")
review_df = spark.read.json(filePath+"/parsed_review.json")
user_df = spark.read.json(filePath+"/parsed_user.json")
business_df = spark.read.json(filePath+"/parsed_business.json")


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Exploring Business DF

# COMMAND ----------

business_df.display()

# COMMAND ----------

# Replacing . with _ to facilitate calling variables in df 
business_df = business_df.toDF(*(col.replace('.', '_') for col in business_df.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Filtering based on multiple conditions

# COMMAND ----------

# Splitting category column into the first 5 categories and dropping categories column
import pyspark.sql.functions as F
cats = F.split(business_df["categories"], ",")

business_cat = business_df.withColumn("cat1", cats.getItem(0))\
.withColumn("cat1", cats.getItem(0))\
.withColumn("cat2", cats.getItem(1))\
.withColumn("cat3", cats.getItem(2))\
.withColumn("cat4", cats.getItem(3))\
.withColumn("cat5", cats.getItem(4)).drop("categories")
business_cat.display()

# COMMAND ----------

# Creating list of relevant categories based on frequency and filtering by restaurants, bars, recreation and social gathering
from pyspark.sql.functions import count, col, when
business_cat.groupBy("cat1").agg(count("business_id").alias("count")).orderBy(col("count").desc() ).display()
business_cat.groupBy("cat2").agg(count("business_id").alias("count")).orderBy(col("count").desc() ).display()
business_cat.groupBy("cat3").agg(count("business_id").alias("count")).orderBy(col("count").desc() ).display()
business_cat.groupBy("cat4").agg(count("business_id").alias("count")).orderBy(col("count").desc() ).display()
business_cat.groupBy("cat5").agg(count("business_id").alias("count")).orderBy(col("count").desc() ).display()

valid_categories = ["Restaurants", "Food", "Shopping", "Nightlife", "Active Life", "Pizza", "Fast Food", "Bars","Coffee & Tea", "Sandwiches",\
                    "Arts & Entertainment", "Mexican","Hotels & Travel", "American (Traditional)","Italian", "Chinese", "Breakfast & Brunch", "Burgers",\
                    "Bakeries", "American (New)", "Specialty Food","Grocery", "Sushi Bars", "Desserts", "Ice Cream & Frozen Yogurt", "Japanese", "Hotels",\
                    "Cafes", "Seafood", "Beer", "Thai", "Vietnamese", "Salad", "Chicken Wings", "Sports Bars", "Diners", "Delis", "Mediterranean", "Pubs",\
                    "Indian","Asian Fusion","Juice Bars & Smoothies", "Barbeque", "Local Flavor","Steakhouses","Greek", "Lounges", "Middle Eastern",\
                    "Korean", "Wine & Spirits", "Caribbean", "Donuts", "Chicken Shop", "French", "Comfort Food", "Latin American", "Breweries", "Tex-Mex",\
                    "Tapas/Small Plates", "Portuguese", "Hookah Bars", "Gastropubs", "Spanish", "Hookah Bars", "Gay Bars", "Cupcakes", "Tapas Bars",\
                    "Chocolatiers & Shops", "Fish & Chips", "African", "Halal", "Waffles", "Tacos", "Creperies", "Beer Bar","Ramen", "Turkish",\
                    "Cheese Shops", "Butcher", "Brazilian", "Buffets", "Vegan","Cocktail Bars","Hot Dogs", "Bagels", "Dim Sum", "Cajun/Creole",\
                    "Cantonese", "Bistros", "Lebanese","Poke", "Falafel","Donairs", "Peruvian", "Empanadas", "Cuban", "Dominican", "Venezuelan",\
                    "Colombian", "Puerto Rican", "Food Stands", "Patisserie/Cake Shop", "Soup", "Custom Cakes", "Noodles", "Food Trucks",\
                    "Fruits & Veggies", "Canadian (New)", "Moroccan"]

business_categ = business_cat.withColumn("isLeisure", when((col("cat1").isin(valid_categories)) | (col("cat2").isin(valid_categories)) | (col("cat3").isin(valid_categories))\
                  | (col("cat4").isin(valid_categories))| (col("cat5").isin(valid_categories)), 1).otherwise(0))

# COMMAND ----------

# Dropping 5 categ cols since they were only used for filtering the businesses
business_filtered = business_categ.drop("cat1","cat2", "cat3", "cat4", "cat5")

# COMMAND ----------

# Checking null values within the dataset 
from pyspark.sql.functions import count, when, isnan, col
business_null_count = business_filtered.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in business_filtered.columns])

# COMMAND ----------

business_null_count.display()

# COMMAND ----------

# Filtering columns with 75% or less null values 
import numpy as np
total_rows = business_filtered.count()
selected_cols = [key for (key, value) in business_null_count.first().asDict().items() if value/total_rows <= 0.75]
# Checking list of removed columns
np.setdiff1d(business_filtered.columns, selected_cols)

# COMMAND ----------

business_df_sel = business_filtered.select(*selected_cols)

# COMMAND ----------

business_df_sel.count()

# COMMAND ----------

# First we're going to filter for those business who didnt have delivery before covid
business_final = business_df_sel.where((col("attributes_RestaurantsDelivery")== False) | (col("attributes_RestaurantsDelivery").isNull()) | (col("attributes_RestaurantsDelivery")== "None")).selectExpr("*")

# Then we're filtering for those business who didnt have takeout before covid
business_final = business_final.where((col("attributes_RestaurantsTakeOut")== False) | (col("attributes_RestaurantsTakeOut").isNull()) | (col("attributes_RestaurantsTakeOut")== "None")).selectExpr("*")

# Filter to get only open businesses 
business_final = business_final.where(col("is_open") == True).selectExpr("*")

# Dropping all 3 columns after filter 
business_final = business_final.drop("attributes_RestaurantsDelivery", "attributes_RestaurantsTakeOut", "is_open")

# COMMAND ----------

business_final.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Handling values on table to create basetable

# COMMAND ----------

# Dropping address column
business_final = business_final.drop("address")

# COMMAND ----------

# Creating dummy of column attributes_Alcohol
business_final.select(col("attributes_Alcohol")).distinct().display()
business_final = business_final.withColumn("alcohol_none", (business_final.attributes_Alcohol == "None") | (business_final.attributes_Alcohol == "u'none'")\
                         | (business_final.attributes_Alcohol == "'none'") | (business_final.attributes_Alcohol.isNull()))\
                               .withColumn("alcohol_none", when(col("alcohol_none")==True,1).otherwise(0))

business_final = business_final.withColumn("alcohol_full_bar", (business_final.attributes_Alcohol == "u'full_bar'") \
                                           | (business_final.attributes_Alcohol == "'full_bar'"))\
                               .withColumn("alcohol_full_bar", when(col("alcohol_full_bar")==True,1).otherwise(0))

business_final = business_final.withColumn("alcohol_beer_and_wine", (business_final.attributes_Alcohol == "u'beer_and_wine'") \
                                           | (business_final.attributes_Alcohol == "'beer_and_wine'"))\
                               .withColumn("alcohol_beer_and_wine", when(col("alcohol_beer_and_wine")==True,1).otherwise(0))

business_final = business_final.drop("attributes_Alcohol")

# COMMAND ----------

business_final.display()

# COMMAND ----------

# Checking if ambience is specified
business_final = business_final.withColumn("ambience", col("attributes_Ambience").contains("True"))\
                       .withColumn("ambience", when(col("ambience")==True,1).otherwise(0)).drop("attributes_Ambience")


# COMMAND ----------

#Bike parking specified
business_final = business_final.withColumn("bikeParking", when(col("attributes_BikeParking")=="True",1).otherwise(0)).drop("attributes_BikeParking")


# COMMAND ----------

# Identify null values in separate column and then substitute True: 1 else 0
business_final = business_final.withColumn("acceptCC_null", when(col("attributes_BusinessAcceptsCreditCards").isNull(),1).otherwise(0))
business_final = business_final.withColumn("acceptsCreditCard", when(col("attributes_BusinessAcceptsCreditCards")=="True",1)\
                                           .otherwise(0)).drop("attributes_BusinessAcceptsCreditCards")


# COMMAND ----------

# Checking if business has parking specified
business_final = business_final.withColumn("businessParking", col("attributes_BusinessParking").contains("True"))\
                       .withColumn("businessParking", when(col("businessParking")==True,1).otherwise(0)).drop("attributes_BusinessParking")

# COMMAND ----------

# Generating dummies for caters
business_final = business_final.withColumn("appointment", when(col("attributes_ByAppointmentOnly")=="True",1)\
                                           .otherwise(0)).drop("attributes_ByAppointmentOnly")


# COMMAND ----------

# Identify GoodForKids null values in separate column and then substitute True: 1 else 0
business_final = business_final.withColumn("goodForKids_null", when(col("attributes_GoodForKids").isNull(),1).otherwise(0))
business_final = business_final.withColumn("goodForKids", when(col("attributes_GoodForKids")=="True",1)\
                                           .otherwise(0)).drop("attributes_GoodForKids")

# COMMAND ----------

# Identify GoodForKids null values in separate column and then substitute True: 1 else 0
business_final = business_final.withColumn("hasTv_null", when(col("attributes_HasTV").isNull(),1).otherwise(0))
business_final = business_final.withColumn("hasTv", when(col("attributes_HasTV")=="True",1)\
                                           .otherwise(0)).drop("attributes_HasTV")

# COMMAND ----------

# Dummy var of outdoor seating
business_final = business_final.withColumn("outdoorSeating", when(col("attributes_OutdoorSeating")=="True",1)\
                                           .otherwise(0)).drop("attributes_OutdoorSeating")

# COMMAND ----------

# Checking restaurant attire values
business_final.groupBy("attributes_RestaurantsAttire").agg(count("business_id").alias("count")).orderBy(col("count").desc() ).display()

#Creating dummy vars
business_final = business_final.withColumn("attire_null", business_final.attributes_RestaurantsAttire.isNull())\
                               .withColumn("attire_null", when(col("attire_null")==True,1).otherwise(0))

# For this variable none provides info, hence categ none is separate 
business_final = business_final.withColumn("attire_none", business_final.attributes_RestaurantsAttire == "None")\
                               .withColumn("attire_none", when(col("attire_none")==True,1).otherwise(0))

business_final = business_final.withColumn("attire_casual", (business_final.attributes_RestaurantsAttire == "u'casual'")\
                                           | (business_final.attributes_RestaurantsAttire == "'casual'"))\
                               .withColumn("attire_casual", when(col("attire_casual")==True,1).otherwise(0))
                                           

business_final = business_final.withColumn("attire_dressy", (business_final.attributes_RestaurantsAttire == "u'dressy'") \
                                           | (business_final.attributes_RestaurantsAttire == "'dressy'"))\
                               .withColumn("attire_dressy", when(col("attire_dressy")==True,1).otherwise(0))
                                           
business_final = business_final.withColumn("attire_formal", (business_final.attributes_RestaurantsAttire == "u'formal'") \
                                           | (business_final.attributes_RestaurantsAttire == "'formal'"))\
                               .withColumn("attire_formal", when(col("attire_formal")==True,1).otherwise(0))
                                                                                      
business_final = business_final.drop("attributes_RestaurantsAttire")

# COMMAND ----------

# Generating dummies for caters
business_final = business_final.withColumn("caters", when(col("attributes_RestaurantsGoodForGroups")=="True",1)\
                                           .otherwise(0)).drop("attributes_RestaurantsGoodForGroups")


# COMMAND ----------

# Checking noise level values
business_final.groupBy("attributes_RestaurantsPriceRange2").agg(count("business_id").alias("count")).orderBy(col("count").desc() ).display()

#Creating dummy vars
business_final = business_final.withColumn("rest_price_range_none", (business_final.attributes_RestaurantsPriceRange2.isNull())\
                                           |(business_final.attributes_RestaurantsPriceRange2 == "None" ))\
                               .withColumn("rest_price_range_none", when(col("rest_price_range_none")==True,1).otherwise(0))

business_final = business_final.withColumn("rest_price_range_1", business_final.attributes_RestaurantsPriceRange2 == "1")\
                               .withColumn("rest_price_range_1", when(col("rest_price_range_1")==True,1).otherwise(0))

business_final = business_final.withColumn("rest_price_range_2", business_final.attributes_RestaurantsPriceRange2 == "2")\
                               .withColumn("rest_price_range_2", when(col("rest_price_range_2")==True,1).otherwise(0))

business_final = business_final.withColumn("rest_price_range_3", business_final.attributes_RestaurantsPriceRange2 == "3")\
                               .withColumn("rest_price_range_3", when(col("rest_price_range_3")==True,1).otherwise(0))

business_final = business_final.withColumn("rest_price_range_4", business_final.attributes_RestaurantsPriceRange2 == "4")\
                               .withColumn("rest_price_range_4", when(col("rest_price_range_4")==True,1).otherwise(0))
                                         
business_final = business_final.drop("attributes_RestaurantsPriceRange2")

# COMMAND ----------

# Dummy to restaurants reservations
business_final = business_final.withColumn("caters", when(col("attributes_RestaurantsReservations")=="True",1)\
                                           .otherwise(0)).drop("attributes_RestaurantsReservations")


# COMMAND ----------

# Checking wifi values
business_final.groupBy("attributes_WiFi").agg(count("business_id").alias("count")).orderBy(col("count").desc() ).display()

#Creating dummy vars
business_final = business_final.withColumn("wifi_none", (business_final.attributes_WiFi.isNull())\
                                           |(business_final.attributes_WiFi == "None" ))\
                               .withColumn("wifi_none", when(col("wifi_none")==True,1).otherwise(0))


business_final = business_final.withColumn("wifi_no", (business_final.attributes_WiFi == "u'no'")\
                                           | (business_final.attributes_WiFi == "'no'"))\
                               .withColumn("wifi_no", when(col("wifi_no")==True,1).otherwise(0))

business_final = business_final.withColumn("wifi_free", (business_final.attributes_WiFi == "u'free'")\
                                           | (business_final.attributes_WiFi == "'free'"))\
                               .withColumn("wifi_free", when(col("wifi_free")==True,1).otherwise(0))

business_final = business_final.withColumn("wifi_paid", (business_final.attributes_WiFi == "u'paid'")\
                                           | (business_final.attributes_WiFi == "'paid'"))\
                               .withColumn("wifi_paid", when(col("wifi_paid")==True,1).otherwise(0))

business_final = business_final.drop("attributes_WiFi")

# COMMAND ----------

# main cities based on freq
main_cities = business_final.groupBy("city").agg(count("business_id").alias("count")).where(col("count")>45).select('city')
main_cities = [row.city for row in main_cities.collect()]

business_final = business_final.withColumn("city_others", (business_final.city.isNull())\
                                           |(~business_final.city.isin(main_cities) ))\
                               .withColumn("city_others", when(col("city_others")==True,1).otherwise(0))

for city in main_cities:
    business_final = business_final.withColumn("city_"+city, business_final.city == city)\
                               .withColumn("city_"+city, when(col("city_"+city)==True,1).otherwise(0))

business_final = business_final.drop("city")

# COMMAND ----------

days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday","Saturday", "Sunday"]

business_final.groupBy("hours_Saturday").agg(count("business_id").alias("count")).orderBy(col("count").desc()).display()

business_final = business_final.withColumn("open_weekends", (~col("hours_Saturday").isNull()) | (~col("hours_Sunday").isNull()) )\
                                .withColumn("open_weekends", when(col("open_weekends")==True,1).otherwise(0))
for dow in days_of_week:
    business_final = business_final.withColumn("miss_sched_"+dow, col("hours_"+dow).isNull())\
                                   .withColumn("miss_sched_"+dow, when(col("miss_sched_"+dow)==True,1).otherwise(0))

    business_final = business_final.withColumn("24h_"+dow, col("hours_"+dow) == "0:0-0:0")\
                                   .withColumn("24h_"+dow, when(col("24h_"+dow)==True,1).otherwise(0))

    business_final = business_final.withColumn("other_sched_"+dow, (col("hours_"+dow) != "0:0-0:0") & (~col("hours_"+dow).isNull()) )\
                                   .withColumn("other_sched_"+dow, when(col("other_sched_"+dow)==True,1).otherwise(0))

    business_final = business_final.drop("hours_"+dow)

# COMMAND ----------

# Dropping unused columns
business_final = business_final.drop("latitude", "longitude","name","postal_code", "state")

# COMMAND ----------

business_final.display(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Exploring Covid DF

# COMMAND ----------

# Checking if table has duplicate values
print(covid_df.count())
print(covid_df.select(col("business_id")).distinct().count())

# COMMAND ----------

covid_df.display(10)

# COMMAND ----------

# Removing duplicates from business id and target variable
covid_no_dup = covid_df.dropDuplicates(["business_id", "delivery or takeout"])

# COMMAND ----------

# Confirming that it doesnt have dup values
covid_no_dup.count()
# Subsetting columns, only business_id and target variable to merge with business table
covid_target = covid_no_dup.select(col("business_id"), col("delivery or takeout"))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Joining Business and Covid Tables

# COMMAND ----------

business_covid  = business_final.join(covid_target,["business_id"],"inner")

# COMMAND ----------

business_covid.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Exploring Checkin DF

# COMMAND ----------

checkin_df.display(10)

# COMMAND ----------

# Getting the date 
from pyspark.sql.functions import to_date
checkin_df = checkin_df.withColumn("date_new",to_date(checkin_df.date))
checkin_df.show(2)

# COMMAND ----------

from pyspark.sql.functions import min, max, count

checkin_df = checkin_df.groupby('business_id').agg(min('date_new').alias('First_date'), 
                                             max('date_new').alias('Last_date'),
                                             count('business_id').alias('checkin_count'))

# COMMAND ----------

from pyspark.sql.functions import col, year, months_between, lit
# Creating recency column from March 2020 to last checkin date in months
checkin_df = checkin_df.withColumn("project_date", lit("2020-03-01"))
checkin_df = checkin_df.withColumn("checkin_recency_months", months_between(to_date(col("project_date")), col("Last_date")))


# COMMAND ----------

# Creating difference between first and last checkin date in years
from pyspark.sql.functions import datediff

checkin_df = checkin_df.withColumn("checkin_date_diff", datediff(col("Last_date"),col("First_date"))/365.25)

# Average of checkins per year 
checkin_df = checkin_df.withColumn("avg_checkin_yr", col("checkin_count")/col("checkin_date_diff"))


# COMMAND ----------

# Drop the columns not useful
col = ("First_date", "Last_date", "project_date")
checkin_df = checkin_df.drop(*col) 

# COMMAND ----------

checkin_df.display(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Merging Checkin with Business-Target

# COMMAND ----------

business_covid.display(2)

# COMMAND ----------

business_covid_checkin  = business_covid.join(checkin_df, ["business_id"],"left")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Exploring Tips DF

# COMMAND ----------

# Getting the date 
from pyspark.sql.functions import to_date
tips_df = tips_df.withColumn("date_tip",to_date(tips_df.date))
tips_df.display(2)

# COMMAND ----------

from pyspark.sql.functions import min, max, count, sum
# Casting compliment count to int 
tips_df = tips_df.withColumn("compliment_count", tips_df.compliment_count.cast("int"))
tips_df_groupped = tips_df.groupby('business_id').agg(min('date_tip').alias('First_Date'), 
                                             max('date_tip').alias('Last_Date'),
                                             count('user_id').alias('users_tip_ct'), sum("compliment_count").alias("total_compliment_ct"))

# COMMAND ----------

from pyspark.sql.functions import months_between, to_date
# Creating recency column from March 2020 to last tip date in months
tips_df_groupped = tips_df_groupped.withColumn("project_date", lit("2020-03-01"))
tips_df_groupped = tips_df_groupped.withColumn("tips_recency_months", months_between(to_date(tips_df_groupped.project_date),tips_df_groupped.Last_Date))

# COMMAND ----------

from pyspark.sql.functions import when

# Creating difference between first and last tip date in years
tips_df_groupped = tips_df_groupped.withColumn("tips_date_diff", datediff(tips_df_groupped.Last_Date,tips_df_groupped.First_Date)/365.25)

# Average of tips per year 
tips_df_groupped = tips_df_groupped.withColumn("avg_tips_yr", tips_df_groupped.users_tip_ct/tips_df_groupped.tips_date_diff)

# If avg tips year is null it means it's only been one tip, therefore the avg will be the users_tip_ct 
tips_df_groupped = tips_df_groupped.withColumn("avg_tips_year",when(tips_df_groupped.avg_tips_yr.isNull(), tips_df_groupped.users_tip_ct).otherwise(tips_df_groupped.avg_tips_yr))
tips_df_groupped = tips_df_groupped.drop("avg_tips_yr")


# COMMAND ----------

# Drop unnecesary columns
col = ("First_Date","Last_Date", "project_date")
tips_df_groupped = tips_df_groupped.drop(*col)

# COMMAND ----------

tips_df_groupped.display(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Merging with previous tables

# COMMAND ----------

business_covid_checkin_tips  = business_covid_checkin.join(tips_df_groupped,["business_id"],"left")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Exploring User and Rating DFs 

# COMMAND ----------

user_df.display(4)

# COMMAND ----------

from pyspark.sql.functions import *
user_df = user_df.withColumn("Total_Recieved_Compliments",  expr("compliment_cool + compliment_cute + compliment_funny + compliment_hot + compliment_list + compliment_more + compliment_note + compliment_photos + compliment_plain + compliment_profile + compliment_writer"))

user_df = user_df.withColumn("Total_Given_Votes",  expr("useful + funny + cool"))

user_df =  user_df.withColumn("Last_Date", lit("2020-07-01 00:00:00"))

user_df =  user_df.withColumn("Loyalty" , (unix_timestamp(to_timestamp("Last_Date")) - unix_timestamp(to_timestamp("yelping_since"))))

user_df =  user_df.withColumn("Loyalty" , round(col("Loyalty")/2629743,3))

user_df = user_df.select('*', split(col("elite"),",").alias("EliteArray"))

user_df = user_df.withColumn("Elite_Years",size(col("EliteArray") )- 1)

user_df = user_df.select('*', split(col("friends"),",").alias("friendsArray"))

user_df = user_df.withColumn("Yelp_Friends",size(col("friendsArray") )- 1)

# COMMAND ----------

user_df = user_df.drop("compliment_cool","compliment_cute","compliment_funny","compliment_hot","compliment_list","compliment_more","compliment_note","compliment_photos","compliment_plain","compliment_profile","compliment_writer","useful","funny","cool","name","Last_Date","elite","friendsArray","friends", "EliteArray" )

# COMMAND ----------

#Only Subsetting the Reviews of Users with atleast 30 reviews
user_df = user_df.where("review_count > 30")

# COMMAND ----------

user_df.display(2)

# COMMAND ----------

review_df.display(2)

# COMMAND ----------

avg_business = review_df.groupBy("business_id").agg(avg("stars").alias("Avg_Ovr_Rating") , count("review_id").alias("Count_Of_Rating"))

avg_business = avg_business.withColumnRenamed("business_id","business_id_2")

avg_business.display(2)

# COMMAND ----------

check_df = review_df.groupBy("business_id","user_id").count()
check_df.where("count > 2").display(5)

# COMMAND ----------

avg_b_u = review_df.groupBy("business_id","user_id").agg(avg("stars").alias("Avg_Per_User_Rating"))

avg_b_u = avg_b_u.withColumnRenamed("business_id","business_id_3")
avg_b_u = avg_b_u.withColumnRenamed("user_id","user_id_2")

# COMMAND ----------

avg_u = review_df.groupBy("user_id").agg(avg("useful").alias("Avg_useful"))
avg_u = avg_u.withColumnRenamed("user_id","user_id_3")

# COMMAND ----------

avg_u.where("Avg_useful > 5").display(5)

# COMMAND ----------

user_df.createOrReplaceTempView("user")
review_df.createOrReplaceTempView("review")
avg_business.createOrReplaceTempView("avg_business")
avg_b_u.createOrReplaceTempView("avg_b_u")
avg_u.createOrReplaceTempView("avg_u")

# COMMAND ----------

# review = spark.sql("select * from review r, avg_business a " +
#     "where r.business_id == a.business_id_2 ")
review = review_df.join(avg_business, review_df["business_id"] == avg_business["business_id_2"], "left")

# COMMAND ----------

# review.createOrReplaceTempView("review")
# review = spark.sql("select * from review r, avg_b_u a " +
#     "where r.business_id == a.business_id_3")
review = review.join(avg_b_u, (review["business_id"] == avg_b_u["business_id_3"]) & (review["user_id"] == avg_b_u["user_id_2"]), "left")

# COMMAND ----------

user = user_df.join(avg_u, user_df["user_id"] == avg_u["user_id_3"], "left")

# COMMAND ----------

review_user = review.join(user, ["user_id"], "left")

# COMMAND ----------

review_user.display()

# COMMAND ----------

review_groupped = review_user.groupby("business_id").agg(sum("cool").alias("cool_reviews"), sum("funny").alias("funny_reviews"),\
                                                    sum("useful").alias("useful_reviews"), avg("Avg_Ovr_Rating").alias("avg_overall_rating"),\
                                                    avg("Count_Of_Rating").alias("avg_rating_count"),\
                                                    avg("Avg_Per_User_Rating").alias("avg_rating_per_user"),\
                                                    avg("average_stars").alias("average_stars_user"), avg("fans").alias("avg_users_fans"),\
                                                    avg("review_count").alias("avg_reviews_per_user"),\
                                                    avg("Total_Recieved_Compliments").alias("avg_users_compliments"),\
                                                   avg("Total_Given_Votes").alias("avg_users_given_votes"), avg("Loyalty").alias("avg_users_loyalty"),\
                                                   avg("Elite_Years").alias("avg_user_elite_yrs"), avg("Yelp_Friends").alias("avg_users_yelp_friends"),\
                                                   avg("Avg_useful").alias("avg_useful_per_user"))

# COMMAND ----------

review_groupped.display(2)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Merging user/review with previous dfs

# COMMAND ----------

base_table = business_covid_checkin_tips.join(review_groupped, ["business_id"], "left").fillna(0).fillna("0")

# COMMAND ----------

base_table = base_table.withColumn("target", when(col("delivery or takeout") == "TRUE", 1).otherwise(0)).drop("delivery or takeout")

# COMMAND ----------

base_table.display(5)

# COMMAND ----------

# MAGIC %md
# MAGIC MODEL BUILDING & TESTING

# COMMAND ----------


##### Loading BaseTable & Train/Test Split

filePath ="/FileStore/tables/complete_base_table.csv"
base_table = spark.read.format("csv")\
                .option("header", "true")\
                .option("multiline", "true")\
                .option("inferSchema", "true")\
                .option("sep", ",")\
                .load(filePath)

base_table.where(col("target")==1).count()

train, test = base_table.randomSplit(weights=[0.8,0.2], seed=200)

test.display()

# COMMAND ----------

##### Logistic Regression

from pyspark.ml.classification import LogisticRegression

logreg = LogisticRegression(family="binomial", standardization= True, featuresCol="features", labelCol="label")
lrModel = logreg.fit(train_set)


#Show information on the final, trained model
summary = lrModel.summary
print(summary.areaUnderROC)
print(summary.accuracy)

# COMMAND ----------

lrModel.coefficients

# COMMAND ----------

###### Predicting on test set

from pyspark.ml.evaluation import BinaryClassificationEvaluator 
lr_pred = lrModel.transform(test_set)
evaluator = BinaryClassificationEvaluator()
print("Test AUC: ", evaluator.evaluate(lr_pred))

accuracy = lr_pred.filter(lr_pred.label == lr_pred.prediction).count() / float(lr_pred.count())
print("Accuracy : ",accuracy)

lr_pred.where(col("prediction")==1.0).count()

# COMMAND ----------

###### Predicting on test set

from pyspark.ml.evaluation import BinaryClassificationEvaluator 
lr_pred = lrModel.transform(test_set)
evaluator = BinaryClassificationEvaluator()
print("Test AUC: ", evaluator.evaluate(lr_pred))

accuracy = lr_pred.filter(lr_pred.label == lr_pred.prediction).count() / float(lr_pred.count())
print("Accuracy : ",accuracy)

lr_pred.where(col("prediction")==1.0).count()

# COMMAND ----------

###### Cross Validation for Params
###### CODE REF: https://medium.com/swlh/logistic-regression-with-pyspark-60295d41221


from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

paramGrid = (ParamGridBuilder()
            .addGrid(logreg.regParam, [0.01,0.1,0.5,2.0])
            .addGrid(logreg.elasticNetParam, [0.0,0.5,1.0])
            .addGrid(logreg.maxIter, [1,5,10])
            .build())
cv = CrossValidator(estimator= logreg, estimatorParamMaps = paramGrid, evaluator = evaluator, numFolds=10)

cvModel = cv.fit(train_set)

# COMMAND ----------

# Evaluate Best Model
pred = cvModel.transform(test_set)
print("Best AUC",evaluator.evaluate(pred))

# COMMAND ----------

#####  Random Forest

from pyspark.ml.classification import RandomForestClassifier

rfClassifier = RandomForestClassifier(numTrees = 10, maxDepth = 10, seed= 123, featuresCol = "features", labelCol = "label")
rfcModel = rfClassifier.fit(train_set)

#Show information on the final, trained model
summary = rfcModel.summary
print(summary.areaUnderROC)
print(summary.accuracy)

# COMMAND ----------

###### Evaluating on test

from pyspark.ml.evaluation import BinaryClassificationEvaluator 
rf_pred = rfcModel.transform(test_set)
evaluator = BinaryClassificationEvaluator()
print("Test AUC: ", evaluator.evaluate(rf_pred))

accuracy = rf_pred.filter(rf_pred.label == rf_pred.prediction).count() / float(rf_pred.count())
print("Accuracy : ",accuracy)

# COMMAND ----------


rfcModel.featureImportances

# COMMAND ----------

# CODE REF: https://www.timlrx.com/blog/feature-selection-using-feature-importance-score-creating-a-pyspark-estimator
import pandas as pd
def ExtractFeatureImp(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))


ExtractFeatureImp(rfcModel.featureImportances, train_set, "features").display()

# COMMAND ----------

###### Feature Selection based on importance

from pyspark.sql.functions import col
important_features = ExtractFeatureImp(rfcModel.featureImportances, train_set, "features")
feats = list(important_features[important_features["score"]>0.01]["name"])
feats

# COMMAND ----------

base_table_imp_feat = base_table.select(*feats, col("target"))
train_imp, test_imp = base_table_imp_feat.randomSplit(weights=[0.8,0.2], seed=200)
basetable_imp = RFormula(formula="target ~ . - business_id").fit(base_table_imp_feat).transform(base_table).select("features","label")
train_imp = RFormula(formula="target ~ . - business_id").fit(train_imp).transform(train_imp).select("features","label")
test_imp = RFormula(formula="target ~ . - business_id").fit(test_imp).transform(test_imp).select("features","label")

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier

rfClassifier = RandomForestClassifier(featuresCol = "features", labelCol = "label")
rfcModel = rfClassifier.fit(train_imp)

#Show information on the final, trained model
summary = rfcModel.summary
print(summary.areaUnderROC)
print(summary.accuracy)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator 
rf_pred_imp = rfcModel.transform(test_imp)
evaluator_imp = BinaryClassificationEvaluator()
print("Test AUC: ", evaluator_imp.evaluate(rf_pred_imp))

accuracy_imp = rf_pred_imp.filter(rf_pred_imp.label == rf_pred_imp.prediction).count() / float(rf_pred_imp.count())
print("Accuracy : ",accuracy_imp)

# COMMAND ----------

###### Cross validation RF tuning params

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

paramGridRF = (ParamGridBuilder()
            .addGrid(rfClassifier.maxDepth, [2,5,10])
            .addGrid(rfClassifier.maxBins, [5,10,20])
            .addGrid(rfClassifier.numTrees, [5,10,20])
            .build())
cv = CrossValidator(estimator= rfClassifier, estimatorParamMaps = paramGridRF, evaluator = evaluator, numFolds=10)

cvModel = cv.fit(train_set)

# COMMAND ----------

# Evaluate Best Model
predRF = cvModel.transform(test_set)
print("Best AUC",evaluator.evaluate(predRF))

best_model_rf = cvModel.bestModel

print("Best Param (maxDepth)", best_model_rf._java_obj.getMaxDepth())
print("Best Param (maxBins)", best_model_rf._java_obj.getMaxBins())
print("Best Param (numTrees)", best_model_rf._java_obj.getNumTrees())

# COMMAND ----------

###### Fitting model with optimal params

rfClassifier_optim = RandomForestClassifier(maxDepth= 10, maxBins=10, numTrees=20, featuresCol = "features", labelCol = "label")
rfcModelOptim = rfClassifier_optim.fit(train_set)

#Show information on the final, trained model
summary_optim = rfcModelOptim.summary
print(summary_optim.areaUnderROC)
print(summary_optim.accuracy)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator 

# Evaluation on test set based on optimal params
rf_pred_optim = rfcModelOptim.transform(test_set)
evaluator = BinaryClassificationEvaluator()

evaluator = BinaryClassificationEvaluator()
auroc = evaluator.evaluate(rf_pred_optim,  {evaluator.metricName: "areaUnderROC"})
print("Test AUC: ", auroc)

accuracy_optim = rf_pred_optim.filter(rf_pred_optim.label == rf_pred_optim.prediction).count() / float(rf_pred_optim.count())
print("Accuracy : ",accuracy_optim)

# COMMAND ----------

rf_pred_optim.show(5)

# COMMAND ----------

import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.plot([0, 1], [0, 1], 'r--')
plt.plot(rfcModelOptim.summary.roc.select('FPR').collect(),
         rfcModelOptim.summary.roc.select('TPR').collect())
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

# COMMAND ----------

###### Extracting test set probability

from pyspark.sql import functions as F
from pyspark.sql.types import FloatType

probability_1 = F.udf(lambda x: float(x[1]), FloatType())
rf_pred_optim.select("probability").select([probability_1(c).alias(c) for c in rf_pred_optim.select("probability").columns]).display()

# COMMAND ----------

##### Gradient Boosting

from pyspark.ml.classification import GBTClassificationModel, GBTClassifier

GBT = GBTClassifier(labelCol="label", featuresCol="features", maxIter=10)

# from pyspark.ml import Pipeline
# pipeline = Pipeline(stages=[GBT])
  
gbtModel = GBT.fit(train_set)
# prediction_train = model.transform(train_set)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator 

# Evaluation on test set based on optimal params
gb_pred_train = gbtModel.transform(train_set)
gb_pred_test = gbtModel.transform(test_set)

evaluator = BinaryClassificationEvaluator()
auroc = evaluator.evaluate(gb_pred_train)
auroc_test = evaluator.evaluate(gb_pred_test)

print("Train AUC: ", auroc)
print("Test AUC: ", auroc_test)

# COMMAND ----------

accuracy_train = gb_pred_train.filter(gb_pred_train.label == gb_pred_train.prediction).count() / float(gb_pred_train.count())
accuracy_test = gb_pred_test.filter(gb_pred_test.label == gb_pred_test.prediction).count() / float(gb_pred_test.count())

print("Accuracy Train: ",accuracy_train)
print("Accuracy Test: ",accuracy_test)

# COMMAND ----------

probability_1 = F.udf(lambda x: float(x[1]), FloatType())
gb_pred_test.select("probability").select([probability_1(c).alias(c) for c in gb_pred_test.select("probability").columns]).display()

rf_pred_optim.where(col("prediction")== 1.0).count()

#test on testing data
prediction = model.transform(test_set)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
binEval = MulticlassClassificationEvaluator().setMetricName("accuracy") .setPredictionCol("prediction").setLabelCol("label")
    
print("Best AUC",binEval.evaluate(prediction)) 

# COMMAND ----------

###### Cross Validation Gradient Boosting

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

paramGridGB = (ParamGridBuilder()
            .addGrid(GBT.maxDepth, [2,5,10])
            .addGrid(GBT.maxBins, [5,10,20])
            .addGrid(GBT.stepSize, [0.1,0.2])
            .build())
cvGB = CrossValidator(estimator= GBT, estimatorParamMaps = paramGridGB, evaluator = evaluator, numFolds=10)

cvModelGB = cvGB.fit(train_set)

predGB = cvModelGB.transform(test_set)
print("Best AUC",evaluator.evaluate(predGB))

best_model_gb = cvModelGB.bestModel

print("Best Param (maxDepth)", best_model_gb._java_obj.getMaxDepth())
print("Best Param (maxBins)", best_model_gb._java_obj.getMaxBins())
print("Best Param (stepSize)", best_model_gb._java_obj.getStepSize())

GBTOptim = GBTClassifier(labelCol="label", featuresCol="features", maxIter=10, maxDepth = 5, maxBins = 10, stepSize = 0.2)
gbtModelOptim = GBTOptim.fit(train_set)

# COMMAND ----------

# Evaluation on test set based on optimal params
gb_pred_train_optim = gbtModelOptim.transform(train_set)
gb_pred_test_optim = gbtModelOptim.transform(test_set)

evaluator = BinaryClassificationEvaluator()
auroc = evaluator.evaluate(gb_pred_train_optim)
auroc_test = evaluator.evaluate(gb_pred_test_optim)

print("Train AUC: ", auroc)
print("Test AUC: ", auroc_test)

accuracy_train = gb_pred_train_optim.filter(gb_pred_train_optim.label == gb_pred_train_optim.prediction).count() / float(gb_pred_train_optim.count())
accuracy_test = gb_pred_test_optim.filter(gb_pred_test_optim.label == gb_pred_test_optim.prediction).count() / float(gb_pred_test_optim.count())

print("Accuracy Train: ",accuracy_train)
print("Accuracy Test: ",accuracy_test)

# Evaluate Best Model
pred = cvModel.transform(test_set)
print("Best AUC",evaluator.evaluate(pred))
