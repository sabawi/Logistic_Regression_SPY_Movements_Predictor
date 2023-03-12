#!/usr/bin/env python
# coding: utf-8

# ## Linear Logistics Models
# ### for 
# ## Predicting Stock Market Movements
# The moves UP or Down are classified only

# In[1]:


"""
By Al Sabawi
2023-03-11 
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import yfinance as yf
import matplotlib.pyplot as plt

# Load SPY stock price data
# df = pd.read_csv('SPY.csv', parse_dates=['Date'], index_col='Date')
df_in = yf.download('SPY',period='max')


# ### Latest SPY ETF Dataset from Yahoo Finance  

# In[2]:


# Create an isolated dataframe so we can manipulate without changing the original data
df = df_in.copy()

# Drop all the columns not relevant for the daily predictions
df = df.drop(columns=['Open','High','Low','Adj Close','Volume'])

# Create a 'Returns' column for the % changes in price and add it to Dataframe
df['ret'] = df['Close'].pct_change()   

# The 'Target' is what we will predict in LogisticRegression 
# The Target is 1 or 0, 1 meaning the stock went up TODAY from Yesterday's price
# However, since we need to predict it a day ahead so we can buy it, we need to shift() back in time!
# so we get the signal to buy before the day the price goes up
## The following line says: If tomorrow's return 'df['ret'].shift(-1)' is above 0, record a buy signal (1) 
# today so we buy it at the open tomorrow, else record 'no buy' signal (0)
df['Target'] = np.where(df['ret'].shift(-1) > 0, 1, 0)


# ### Creating LAGGED Dataset

# In[3]:


# A lagged dataset in Timeseries is based on the assumption that the predicted value 'Target' 
# depends on the prices of 1 or more days before.  In this case I am taking into account 5 days before
# We will add 5 new columns recording the change in price for the past 5 days in each row

# Create lagged features for the past 5 days
def create_lagged_features(df, lag):
    features = df.copy()
    for i in range(1, lag+1):
        features[f'ret_lag{i}'] = features['ret'].shift(i)
    features.dropna(inplace=True)
    features.drop(columns=['Close'],inplace=True)
    features.drop(columns=['ret'],inplace=True)
    return features

df_lagged = create_lagged_features(df, 5)
df_lagged.tail(6)


# ### Training Set and Batches
# ##### We'll need to divide the historical data in smaller batches but we need to make sure each batch is balanced as much as possible

# In[4]:


# Split data into train and test sets using a stratified 80-20 split
df_lagged.dropna(inplace=True)
train_df, test_df = train_test_split(df_lagged, test_size=0.2, random_state=42, stratify=df_lagged['Target'])

# ##############################################################
# About Batches:    For a LogisticRegression Model, we need to 
#                   balance the training data with rows that have 
#                   equal 'Target' of 1 (buy) and 0 (no buy). 
#                   Otherwise the model will become bias for the outcome
#                   that we feed it more of.  So for that we made each 
#                   row 'self-contained' with all the previous 
#                   data (last return plu 5 previous returns) so that 
#                   we can shuffle the rows and feed them into 
#                   the model as batches of rows. Each bach is an equal 
#                   mix of outcome 1 and 0.  This was the concentration 
#                   of 1's and 0's dont in a series (long up trends or 
#                   long down trends) don't bias the next outcome
# ###############################################################

# Split train data into batches with balanced target values
batches_count = 128  # We can start from 32 to go up then see the accuracy the effect of accuracy
batch_size = len(train_df) // batches_count
train_batches = []
for i in range(0, len(train_df), batch_size):
    batch = train_df.iloc[i:i+batch_size]
    num_positives = len(batch[batch['Target'] == 1])
    if num_positives == batch_size // 2:
        train_batches.append(batch)
    elif num_positives > batch_size // 2:
        excess_positives = num_positives - batch_size // 2
        batch = batch.drop(batch[batch['Target'] == 1].sample(excess_positives).index)
        train_batches.append(batch)
    else:
        missing_positives = batch_size // 2 - num_positives
        num_negatives = len(batch[batch['Target'] == 0])
        if missing_positives > num_negatives:
            batch = batch.drop(batch[batch['Target'] == 0].index)
            missing_positives -= num_negatives
            excess_positives = missing_positives - len(batch[batch['Target'] == 1])
            batch = pd.concat([batch, batch[batch['Target'] == 1].sample(excess_positives, replace=True)])
        else:
            batch = batch.drop(batch[batch['Target'] == 0].sample(missing_positives, replace=False).index)
        train_batches.append(batch)
        
print(f"Number of Batches = {batches_count}")
print(f"Rows in first batch = {len(train_batches[0])}")
train_batches[0].tail(len(train_batches[0]))


# In[5]:


# train_df dataframe is the unbatched dataset
train_df.tail(5)


# ### model_1 : Create the 1st Model

# In[6]:


# Create logistic regression model
model_1 = LogisticRegression(class_weight='balanced')


# ### Testing the model (model_1)

# In[7]:


# Train model on the first batch of the training data
X_train = train_batches[0].drop(columns=['Target'])
y_train = train_batches[0]['Target']
# print("X_train: \n",X_train.tail(5))
# print("y_train: \n",y_train.tail(5))
# ******************************************************
model_1.fit(X_train, y_train)

# Evaluate the model on the test set and print the test accuracy
X_test = test_df.drop(columns=['Target'])
y_test = test_df['Target']
y_pred = model_1.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f'Test accuracy: {test_accuracy:.3f}')


# In[8]:


# Show predictions
y_pred


# ### Train the remaining batches on model_1

# In[9]:


print("BatchSize",batch_size)
for j in range(1,batches_count):
    # print("Batch#",j,train_batches[j])
    X_train = train_batches[j].drop(columns=['Target'])
    # print(X_train.columns)
    y_train = train_batches[j]['Target']
    model_1.fit(X_train, y_train)

# X_train = train_df.drop(columns=['Target'])
# y_train = train_df['Target']

# #***************************
# model.fit(X_train, y_train)
# X_train


# ### Predictions from model_1

# In[10]:


# Make predictions for next 5 days
first_value = df_in['Close'][0]
df_pred = df_in.copy()
print(first_value)


# ### Use the whole dataset and removing the Non-Feature columns
# Non-feature columns are columns not used for training

# In[11]:


# Generate the Retuen columns and the Target column to compare with later
df_pred['ret'] = df_pred['Close'].pct_change() # Daily return
df_pred['Target'] = np.where(df_pred['ret'].shift(-1) > 0.0, 1, 0) # Target column is 1 IF the next day close price is higher
df_pred = create_lagged_features(df_pred, 5) # Create the Lagged columns for the past 5 days

# df_pred.drop(columns=['Target'],inplace=True)
df_pred = df_pred.drop(columns=['Open','High','Low','Adj Close','Volume']) # Remove non-feature columns
if 'Predicted' in df_pred.columns:
    df_pred = df_pred.drop('Predicted') # Remove the predicted column in case its leftover from previous runs
df_pred.tail(5)


# In[12]:


# Create a separate dataframe WITHOUT target column for prediction only
data_no_target = df_pred.copy()
if('Target' in data_no_target.columns):
    data_no_target = data_no_target.drop(columns=['Target'])

# Check to see if we have the right no. of columns for the prediction call
print('column count =',len(data_no_target.columns),':',data_no_target.columns)
# Check that we have the Target data still available
df_pred['Target'].tail(5)


# In[13]:


predictions_1 = model_1.predict(data_no_target)


# In[14]:


# Make predictions from model_1 into a DataFrame along with the actual Target column from before to compare
df_pred1 = pd.DataFrame(index=df_pred.index)
df_pred1['Predicted']=  predictions_1
df_pred1['Target'] = df_pred['Target'].copy()
df_pred1[['Predicted','Target']].tail(50)


# ### Check Prediction Results for model_1

# In[15]:


eq=neq=pup=tup=pdown=tdown=0
for i in range(len(df_pred1['Predicted'])):
    if df_pred1['Predicted'].iloc[i] == df_pred1['Target'].iloc[i]:
        eq+=1
        if df_pred1['Predicted'].iloc[i] == 1:
            pup+=1
            tup+=1
        else:
            pdown+=1
            tdown+=1
    else:
        neq+=1
        if df_pred1['Predicted'].iloc[i] == 1:
            pup+=1
            tdown+=1
        if df_pred1['Target'].iloc[i] == 1:
            pdown+=1
            tup+=1
      

print("----Results from Predictions using model_1----")  
print(f"Equal Values = {eq} ({round(100*eq/(eq+neq),2)}%) \n\
Not Equal = {neq} ({round(100*neq/(eq+neq),2)}%),  \n\
Total = {eq+neq} rows")
print(f"Predicted UPs : {round(100*pup/(eq+neq),2)}% vs Actual UPs : {round(100*tup/(eq+neq),2)}%  ")
print(f"Predicted Downs : {round(100*pdown/(eq+neq),2)}% vs Actual Downs : {round(100*tdown/(eq+neq),2)}%  ")


# ### model_2: Creating the second model
# This model will be trained without batches or manual re-balancing of outcomes 

# In[16]:


# Creating model_2 and training it on the whole dataset on one go. No batching or rebalancing 
model_2 = LogisticRegression(class_weight='balanced')
print('Check columns : ',data_no_target.columns)
model_2.fit(data_no_target, df_pred['Target'])
    
df_pred2 = pd.DataFrame(index=df_pred.index)
df_pred2['Predicted']=  model_2.predict(data_no_target)
df_pred2['Target'] = df_pred['Target'].copy()
df_pred2[['Predicted','Target']].tail(50)


# In[17]:


# Reassembling the original dataset with the Predicted and Target columns added
df_in2 = df_in.copy()
df_in2['Predicted Buy'] = df_pred2['Predicted']
df_in2['Correct Buy'] = df_pred2['Target']
df_in2.dropna()
df_in2.tail(20)
# df_pred['Target'].tail(50)


# In[18]:


df_in2.tail(5)


# ### Check Prediction Results for model_2

# In[19]:


eq=neq=pup=tup=pdown=tdown=0
for i in range(len(df_pred1['Predicted'])):
    if df_pred1['Predicted'].iloc[i] == df_pred1['Target'].iloc[i]:
        eq+=1
        if df_pred1['Predicted'].iloc[i] == 1:
            pup+=1
            tup+=1
        else:
            pdown+=1
            tdown+=1
    else:
        neq+=1
        if df_pred1['Predicted'].iloc[i] == 1:
            pup+=1
            tdown+=1
        if df_pred1['Target'].iloc[i] == 1:
            pdown+=1
            tup+=1
      

print("----Results from Predictions using model_2----")  
print(f"Equal Values = {eq} ({round(100*eq/(eq+neq),2)}%) \n\
Not Equal = {neq} ({round(100*neq/(eq+neq),2)}%),  \n\
Total = {eq+neq} rows")
print(f"Predicted UPs : {round(100*pup/(eq+neq),2)}% vs Actual UPs : {round(100*tup/(eq+neq),2)}%  ")
print(f"Predicted Downs : {round(100*pdown/(eq+neq),2)}% vs Actual Downs : {round(100*tdown/(eq+neq),2)}%  ")


# ### Predicting the Stock Market for the next 5 Days 
# We'll use model_2 and follow the same procedures of no batches or re-balancing 

# In[20]:


df_pred = df_in.copy()
# df_pred['Close'].pct_change()
df_pred['ret'] = df_pred['Close'].pct_change()
df_pred['Target'] = np.where(df_pred['ret'].shift(-1) > 0.0, 1, 0)
df_pred = create_lagged_features(df_pred, 5)
df_pred


# In[21]:


df_pred = df_pred.drop(columns=['Open','High','Low','Adj Close','Volume'])
df_pred


# ### 5-Days in the future stock predictions

# In[22]:


# We need al least 5 days from the past without the Target column
last_five_days = df_pred.iloc[-5:].copy()
last_five_days.drop('Target',inplace=True,axis=1)

# We need to add the Predicted column for future predictions
new_columns = last_five_days.columns[-5:].to_list()
new_columns.append('model_1_Predicted')
new_columns.append('model_2_Predicted')

# We need to prepare an empty dataframe to receive future data
next_five_days = pd.DataFrame(columns=new_columns)

# Now starting from the first of the last 5 days, predict tomorrow Up or Down market, then move forward one day
for i in range(1, 6):
    next_day_m1 = model_1.predict(last_five_days.iloc[[i-1]])
    next_day_m2 = model_2.predict(last_five_days.iloc[[i-1]])
    # next_day = model.predict(last_five_days.iloc[i-1, 1:].values.reshape(1, -1))
    arr_m1 = np.append(last_five_days.iloc[i-1, :].values, next_day_m1[0])
    arr_m2 = np.append(arr_m1,next_day_m2[0])
    arr_df = pd.DataFrame([arr_m2], columns=new_columns)
    next_five_days= pd.concat([next_five_days,arr_df])
    
# Create the next 5 working dates and make an index for the predicted 5 days
import datetime
from pandas.tseries.offsets import BDay
daysdates = [(datetime.datetime.today() + BDay(i)).strftime("%Y-%m-%d")  for i in range(1,6) ]
df_next5days = pd.DataFrame(next_five_days)
df_next5days.index = daysdates
print('\nPredictions for next 5 days:')
print(df_next5days[['model_1_Predicted','model_2_Predicted']])


# In[23]:


# Based on the above, Model_1 is predicting a BUY signal at End of day 2023-03-15, which means to BUY the Open on 2023-03-16
# While Model_2 is predicting to BUY the morning Open the next trading day to 2023-03-17 which 2023-03-20

