#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv(r'C:\Users\a.alsakkaf\Desktop\Saydah Al-kaf_file_Surrey\JAD\Youtube_Analysis_project/UScomments.csv', on_bad_lines='skip')

# Display the first few rows of the dataframe
df.head()


# In[3]:


df.isnull()


# In[4]:


df.isnull().sum()


# In[5]:


df.dropna(inplace=True)


# In[6]:


df.isnull().sum()


# In[7]:


get_ipython().system('pip install textblob')


# In[8]:


from textblob import TextBlob


# In[9]:


df.head(6)


# In[10]:


TextBlob("Logan Paul it's yo big day ‼️‼️‼️")


# In[11]:


TextBlob("Logan Paul it's yo big day ‼️‼️‼️").sentiment


# In[12]:


TextBlob("Logan Paul it's yo big day ‼️‼️‼️").sentiment.polarity


# In[13]:


df.shape


# In[14]:


sample_df=df[1:1000]


# In[15]:


sample_df.shape


# In[16]:


polarity=[]
for comment in df['comment_text']:
    try:
        polarity.append(TextBlob(comment).sentiment.polarity)
    except:
         polarity.append(0)
    


# In[17]:


len(polarity)


# In[18]:


df["polarity"]= polarity


# In[19]:


df.head(5)


# In[20]:


filter1=df["polarity"]==1


# In[21]:


df_positive=df[filter1]


# In[22]:


filter2=df["polarity"]==-1


# In[23]:


df_negative=df[filter2]


# In[24]:


df_negative.head(5)


# In[25]:


df_positive.head(5)


# In[26]:


get_ipython().system('pip install wordcloud')


# In[27]:


from wordcloud import WordCloud, STOPWORDS


# In[28]:


set(STOPWORDS)


# In[29]:


df["comment_text"]


# In[30]:


type(df["comment_text"])


# In[31]:


totel_comments_positive=''.join(df_positive["comment_text"])


# In[32]:


wordcloud=WordCloud(stopwords=set(STOPWORDS)).generate(totel_comments_positive)


# In[33]:


plt.imshow(wordcloud)
plt.axis('off')


# In[34]:


totel_comments_negaive=''.join(df_negative["comment_text"])
#total_comments_negative = ''.join(df_negative["comment_text"].astype(str))


# In[35]:


wordcloud2=WordCloud(stopwords=set(STOPWORDS)).generate(totel_comments_negaive)
#wordcloud2 = WordCloud(stopwords=set(STOPWORDS)).generate(total_comments_negative)


# In[36]:


plt.imshow(wordcloud2)
plt.axis('off')


# In[37]:


get_ipython().system('pip install emoji==2.2.0')


# In[38]:


import emoji


# In[39]:


emoji.__version__


# In[40]:


df["comment_text"].head(6)


# In[41]:


comment=' trending 😉'


# In[42]:


[char for char in comment if char in emoji.EMOJI_DATA]


# In[43]:


emoji_list=[]
for char in comment:
    if char in emoji.EMOJI_DATA:
        emoji_list.append(char)


# 

# In[44]:


emoji_list


# In[45]:


all_emoji_list=[]
for char in df["comment_text"].dropna():
    if char in emoji.EMOJI_DATA:
        all_emoji_list.append(char)


# In[46]:


all_emoji_list[0:10]


# In[47]:


from collections import Counter


# In[48]:


Counter(all_emoji_list).most_common(10)


# In[49]:


Counter(all_emoji_list).most_common(10)[0]


# In[50]:


Counter(all_emoji_list).most_common(10)[0][0]


# In[51]:


Counter(all_emoji_list).most_common(10)[0][1]


# In[52]:


Counter(all_emoji_list).most_common(10)[1][0]


# In[53]:


Counter(all_emoji_list).most_common(10)[2][0]


# In[54]:


emojis=[Counter(all_emoji_list).most_common(10)[i][0]for i in  range(10)]


# In[55]:


freqs=[Counter(all_emoji_list).most_common(10)[i][0]for i in  range(10)]


# In[56]:


Counter(all_emoji_list).most_common(10)[0][1]


# In[57]:


Counter(all_emoji_list).most_common(10)[1][1]


# In[58]:


Counter(all_emoji_list).most_common(10)[2][1]


# In[59]:


freqs=[Counter(all_emoji_list).most_common(10)[i][1]for i in  range(10)]
freqs


# In[60]:


import plotly.graph_objs as go
from plotly.offline import iplot


# In[61]:


trace =go.Bar(x=emojis,y= freqs)


# In[62]:


iplot([trace])

import os
os.listdir(r'C:\Users\a.alsakkaf\Downloads\Youtube_project_shan_singh_Udemy\additional_data')
# In[63]:


import os
files=os.listdir(r'C:\Users\a.alsakkaf\Downloads\Youtube_project_shan_singh_Udemy\additional_data')
files


# In[64]:


files_csv= [file for file in files if '.csv' in file]
files_csv


# In[65]:


import warnings
from warnings import filterwarnings
filterwarnings('ignore')


# In[66]:


#full_df=pd.DataFrame()
#path=(r'C:\Users\a.alsakkaf\Downloads\Youtube_project_shan_singh_Udemy\additional_data')
#for file in files_csv:
  #  current_df= pd.read_csv(path+'/'+file ,encoding='iso-8859-1', error_bad_lines= False)
   # pd.concat([full_df,current_df], ignore_index=True)


# In[67]:


full_df = pd.DataFrame()
path = r'C:\Users\a.alsakkaf\Downloads\Youtube_project_shan_singh_Udemy\additional_data'

# Assuming 'files_csv' is a list of csv filenames
for file in files_csv:
    current_df = pd.read_csv(path + '/' + file, encoding='iso-8859-1', on_bad_lines='skip')
    full_df = pd.concat([full_df, current_df], ignore_index=True)


# In[68]:


full_df.shape


# In[69]:


full_df[full_df.duplicated()].shape


# In[70]:


full_df=full_df.drop_duplicates()
full_df.shape


# In[71]:


full_df[1:1000].to_csv(r'C:\Users\a.alsakkaf\Desktop\Youtube_project_shan_singh_Udemy/youtube_sample.csv', index= False)


# In[73]:


#full_df[1:1000].to_json(r'C:\Users\a.alsakkaf\Desktop\Youtube_project_shan_singh_Udemy/youtube_sample.json', index= False)
full_df[1:1000].to_json(r'C:\Users\a.alsakkaf\Desktop\Youtube_project_shan_singh_Udemy/youtube_sample.json', orient='split', index=False)


# In[ ]:





# In[76]:


from sqlalchemy import create_engine


# In[79]:


engine = create_engine(r'sqlite:///C:\Users\a.alsakkaf\Desktop\Youtube_project_shan_singh_Udemy/youtube_sample.sqlite')


# In[80]:


full_df[0:1000].to_sql('Users', con=engine, if_exists='append')


# In[81]:


full_df.head(5)


# In[82]:


full_df['category_id'].unique()


# In[84]:


json_df = pd.read_json(r'C:\Users\a.alsakkaf\Desktop\Youtube_project_shan_singh_Udemy\additional_data/US_category_id.json')


# In[85]:


json_df


# In[86]:


json_df['items']


# In[87]:


json_df['items'][0]


# In[89]:


json_df['items'][1]


# In[91]:


cat_dict={}
for item in json_df['items'].values:
    cat_dict[int(item['id'])] = item['snippet']['title']


# In[92]:


cat_dict


# In[96]:


full_df['category_name'] = full_df['category_id'].map(cat_dict)


# In[97]:


full_df.head(4)


# In[104]:


plt.figure(figsize=(12, 8))
sns.boxplot(x='category_name', y='likes', data=full_df)
plt.xticks(rotation ='vertical')
plt.show()


# In[106]:


full_df['likes_rate']=(full_df['likes']/full_df['views'])*100
full_df['dislikes_rate']=(full_df['dislikes']/full_df['views'])*100
full_df['comment_count_rate']=(full_df['comment_count']/full_df['views'])*100


# In[107]:


full_df.columns


# In[108]:


plt.figure(figsize=(12, 8))
sns.boxplot(x='category_name', y='likes_rate', data=full_df)
plt.xticks(rotation ='vertical')
plt.show()


# In[109]:


sns.regplot( x= 'views', y='likes', data=full_df)


# In[110]:


full_df.columns


# In[111]:


full_df[['views', 'likes', 'dislikes']]


# In[112]:


full_df[['views', 'likes', 'dislikes']].corr()


# In[114]:


sns.heatmap(full_df[['views', 'likes', 'dislikes']].corr(), annot= True)


# In[115]:


full_df.head(6)


# In[116]:


full_df['channel_title'].value_counts()


# In[118]:


full_df.groupby(['channel_title']).size().sort_values()


# In[120]:


cdf = full_df.groupby(['channel_title']).size().sort_values(ascending =False).reset_index()


# In[121]:


cdf


# In[131]:


cdf= cdf.rename(columns={0:'totel videos'})


# In[135]:





# In[140]:


cdf


# In[133]:


import plotly.express as px


# In[138]:


px.bar(data_frame=cdf[0:20] , x ='channel_title', y= 'totel videos')


# In[141]:


full_df["title"]


# In[142]:


full_df["title"][0]


# In[143]:


import string 


# In[144]:


string.punctuation


# In[145]:


[char for char in full_df["title"][0]if char in string.punctuation] 


# In[146]:


len([char for char in full_df["title"][0]if char in string.punctuation])


# In[147]:


def punc_count(text):
    return len([char for char in text if char in string.punctuation])


# In[149]:


full_df['title'].apply(punc_count)


# In[150]:


sample= full_df[0:1000]


# In[151]:


sample ['count_punc'] = sample['title'].apply(punc_count)


# In[152]:


sample ['count_punc'] 


# In[154]:


plt.figure(figsize=(12, 8))
sns.boxplot(x='count_punc', y='views', data=sample)
#plt.xticks(rotation ='vertical')
plt.show()


# In[156]:


plt.figure(figsize=(8, 6))
sns.boxplot(x='count_punc', y='likes', data=sample)
#plt.xticks(rotation ='vertical')
plt.show()


# In[ ]:




