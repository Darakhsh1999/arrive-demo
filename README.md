# Arrive cast study


## Task

Create a presentation for 2 groups of people
- Technical team (lower level details of code and statistical analysis)
- Stakeholders (high-level overview of findings in data)

## Background

- Heterogeneous data
    * parking behaviour
    * different data types
    * finance data

- Various data sources need to be integrated
    * app event data
    * transaction data
    * geolocation data
    * This data exists in its raw form and transformed form in the data warehouse (Snowflake)
    
- Typical tasks for this role
    * Data analysis
    * Knowledge discovery
    * Data modeling
    
- Data Scientist role
    * conception to production
    * understanding business problem
    * data availability
    * model construction
    * model evaluation
    * model deployment
    * model monitoring


## Business case study

NOTE: You do not necessarily need to implement everything you have in mind; instead, you may describe parts of it on a conceptual level


The service is provided to both **individual** users and **businesses** users. Individual users have a private account in the backend system. Business users are tied to a company, that are parking for business reasons. Registered business users have corporate accounts and optionally a private account as well. 


To acquire new business users, we try to identify private users that are parking for business purposes. These users will be contacted by the sales team to convert them to corporate accounts.

The identification process can be supported by various lead sources, such as 
- checking their phone number, email address, registered car (requires 3rd party data support)
- identify parking patterns indicating that a user is parking for business purposes. This can be done by analyzing:
    * transaction data
    * parking duration
    * parking frequency
    * parking location

### Main task

Your task is to create or conceptualize a model using the provided data. The model should predict which private users are parking for business purposes based on their parking behavior

### Data

In the provided data file (assignment_sample_data.csv), there are 87,489 rows and 11 columns.
Each row represents a single parking transaction by a user. The 11 columns are:
- parking_id: a unique id for each parking
- area_type: the type of area where the car was parked.
- parking_start_time: time stamp when the parking started
- parking_end_time: time stamp when the parking ended
- parking_fee: total parking fee paid by the user
- currency: currency of the parking fee
- parkinguser_id: a unique id for the user associated with the parking
- car_id: a unique id for the car associated with the parking
- lat: latitude of the parking location
- lon: longitude for the parking location
- account_type: type of account associated with the user (private or corporate)

### Solution outline

In your solution, we expect to see:
- Problem scoping
- Plan for feature engineering
- Model selection & motivation
- Preliminary insights from data
- Potential limitations & challenges
- Implementation plan
- Other approaches and future work


## Solution plan

- Feature engineer the parking behavior of the users.
    * parking near residential area is an indicator of private use case.
    * parking frequency, many parking transactions per day is indicator of business use case (frequency distribution w.r.t account type)
    * higher parking frequency during weekdays is indicator of business use case while parking during weekends is indicator of private use case
    * car model, some car models are more likely to be used for business purposes (compact car, SUV) while other are more likely to be used for private purposes (sports car, luxury car), unfortunately this information is not available in the data since we cant relate car_id to model
    

Filter 2 dataframes, one for private users and one for corporate users. See if some individauls are closer to business users, if so, what are the features that make them closer to business users?

## TODO

- [ ] change weekday from 0-6 to 0 or 1 where 0 is weekday and 1 is weekend
- [ ] User aggregation
- [ ] Implement a supervised learning model (KNN and Random Forest) perform prediction CV
- [ ] Train a model on both data. Check logits on users labeled as  private users and separate lower confidence users, these should indicate business use case since model's prediction is not aligned with the true label