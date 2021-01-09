##-----Required Libraries ---###
import numpy as np 
import pandas as pd 
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
import hashlib
import sqlite3 



data = pd.read_csv('train.csv')
CatVariables = data.select_dtypes(include=['object'])
NumVariables = data.select_dtypes(include=['float','int'])
#standardizing column names for easier usage
data.columns=data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
CatVariables=CatVariables.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
NumVariables=NumVariables.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
#drop unwanted variables
data = data.drop(columns=['loan_id','dependents']).dropna()
CatVariables = CatVariables.drop(['loan_id','dependents']).dropna()

#more Preprocessing steps
#This not filling the the nulls correct it
#For quantitative data
from sklearn.impute import SimpleImputer
numerical_cols = data[["loan_amount_term"]]
categorical_cols = data[['gender','self_employed']]
imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
lat = imp_mean.fit_transform(numerical_cols)
imp_mode = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
g_se = imp_mode.fit_transform(categorical_cols)

data['loan_amount_term'] = lat
data[['gender','self_employed']] = g_se

# print(f'Your data has {data.isnull().sum()} cleaner')
#Dealing with categorical columns
#Label Encoding for object to numeric conversion
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for feat in CatVariables:
    data[feat] = le.fit_transform(data[feat].astype(str))





## Prediction
X = data[['gender', 'married', 'applicantincome', 'loanamount', 'credit_history']]
y = data.loan_status

# print((X.shape, y.shape))
from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size = 0.2, random_state = 10)

from sklearn.ensemble import RandomForestClassifier 
model = RandomForestClassifier(max_depth=4, random_state = 10) 
model.fit(x_train, y_train)

from sklearn.metrics import accuracy_score
pred_cv = model.predict(x_cv)
print(f'Your train accuracy is: {accuracy_score(y_cv,pred_cv)}')

pred_train = model.predict(x_train)
print(f'Your prediction is: {accuracy_score(y_train,pred_train)}')

# saving the model 
import pickle 
pickle_out = open("classifier.pkl", mode = "wb") 
pickle.dump(model, pickle_out) 
pickle_out.close()

# loading the trained model
pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in)
 
@st.cache()
  
# defining the function which will make the prediction using the data which the user inputs 
def prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History):   
 
    # Pre-processing user input    
    if Gender == "Male":
        Gender = 0
    else:
        Gender = 1
 
    if Married == "Unmarried":
        Married = 0
    else:
        Married = 1
 
    if Credit_History == "Unclear Debts":
        Credit_History = 0
    else:
        Credit_History = 1  
 
    LoanAmount = LoanAmount / 1000
 
    # Making predictions 
    prediction = classifier.predict( 
        [[Gender, Married, ApplicantIncome, LoanAmount, Credit_History]])
     
    if prediction == 0:
        pred = 'Rejected'
    else:
        pred = 'Approved'
    return pred


#set up the UI
PAGE_CONFIG = {"page_title":"Loanstatus.io","page_icon":":smiley:","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)
# Security
#passlib,hashlib,bcrypt,scrypt
def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False
# DB Management
conn = sqlite3.connect('data.db')
c = conn.cursor()
# DB  Functions
def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data


def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data

def main():
	st.markdown("![somestuff](https://az712634.vo.msecnd.net/content/14b2744cf8d6418c87ffddc3f3127242/9502630827244d60a1214f250e3bbca7/073d9d18570e437e9c3a091aa0dc3dae/2d7a9f7214f0450885c9b52f5fbec5fe/image)")
	html_temp = """ 
	<div style ="background-color:#010203;padding:3px"> 
	<h1 style ="color:gold;text-align:center;">Predicting Loan Credit History and Status</h1> 
	<h2 style = "color:aqua; text-align: center;">
	The loan application will visualize the good or bad 
	credit risks according to the set of attributes of bank members.</h2>
	</div> 
	"""

		# display the front end aspect
	st.markdown(html_temp, unsafe_allow_html = True) 
	
	html_temp1 = """ 
	<div style ="background-color:#00ff98;padding:53px"> 
	<p style ="color:purple;text-align:left;">This User Interface is meant to bring to you the members loan status and credit history.\
		 Feel free to navigate through all the sections of this web application.\
		  You can also upload your own data for your country and visualize it. We're all yours.</p> 
	</div> 
	"""

	menu = ["Home","DataFrame", "Visualize","About", "Login","SignUp","Prediction"]
	#If one chooces to proceed with the Home Button --->
	choice = st.sidebar.selectbox('Menu',menu)
	if choice == 'Home':
		st.subheader("Welcome Dear Reader):-")
		st.subheader("Let's Visualize the Loan Prediction Dataset:")
		# display the front end aspect
		st.markdown(html_temp1, unsafe_allow_html = True) 
		st.markdown("")
		st.markdown('Please answer this question to get updates')
		option = st.selectbox( 'How would you like to be contacted?', ('Email', 'Home phone', 'Mobile phone'))
		# st.markdown("To view more, please proceed to the Visualize section under the dropdown menu. It's Fantastic")
		# st.sidebar.title("Visualization Selector")
		# st.sidebar.markdown("Select the Charts/Plots accordingly:")
	#If interested to see the Dataframe ---->
	elif choice == 'DataFrame':
		st.subheader('The data has 100 entries and 17 attributes.')
		st.write('Choose one to continue with the following section:')
		a = st.radio('Choose whole or Subset to continue:',['whole','subset'],1)
		if a == "whole":
			st.dataframe(data.style.set_properties(
        **{"background-color": "lawngreen", "color": "blue"}))
		else:
			b = st.radio('Choose from the options, categorical, numerical, or top 5'\
				,['CatVariables', 'NumVariables', 'top_5'],1)
			if b == 'CatVariables':
				st.dataframe(CatVariables.style.set_properties(
        **{"background-color": "lawngreen", "color": "black"}).highlight_max(axis=0))
			elif b == 'NumVariables':
				st.dataframe(NumVariables.style.set_properties(
        **{"background-color": "lawngreen", "color": "black"}).highlight_max(axis=0))
			else:
				st.dataframe(data.head().style.set_properties(
        **{"background-color": "lawngreen", "color": "black"}).highlight_max(axis=0))

	#if choice made is to see visualizations --->
	elif choice == 'Visualize':
		a = st.radio('Please choose on of the following',['Pie','Bar','Line'],2)
		if a == 'Pie':
			fig = px.pie(data,values=data['credit_history'].value_counts(),\
				names=['Good','Bad'],title='Distribution of Member Credit History, (good or bad)',\
				hole=0.2)
			fig1 = px.pie(data, values = data['loan_status'].value_counts(), \
				names=['Yes','No'],title='Distribution of Member Loan Status, (Yes or No)',\
				hole=0.2)
			st.plotly_chart(fig)
			st.plotly_chart(fig1)
		if a == 'Bar':
		# 	st.title("Bar Charts of Cases,Deaths and Recoveries")	
			fig = go.Figure(data=[
			go.Bar(name='Gender vs Income',x=data['gender'], y=data['applicantincome']),
			go.Bar(name = 'Self Employement vs Savings',x=data.self_employed, y=data['totalsavings']),
		# 	go.Bar(name='South Korea', x=sk_df.index, y=sk_df['Recovered']),
		# 	go.Bar(name='Georgia', x=geo_df.index, y=geo_df['Recovered'])

			])
			st.plotly_chart(fig)

		# 	st.write('Equally, the bar graph could easily depict the death Rate.')
		# if a == 'Line':
		# 	fig = go.Figure()
		# 	fig.add_trace(go.Scatter(x=uk_df.index, y=uk_df['ConfirmedCases'],name='United Kingdom Deaths'))
		# 	fig.add_trace(go.Scatter(x=swe_df.index, y=swe_df['ConfirmedCases'],name='Sweden Deaths'))
		# 	fig.add_trace(go.Scatter(x=sk_df.index, y=sk_df['ConfirmedCases'],name='South Korea Deaths'))
		# 	fig.add_trace(go.Scatter(x=geo_df.index, y=geo_df['ConfirmedCases'],name='Georgia Deaths'))
			
		# 	fig.update_layout(title="Case Count in the 4 Countries")
		# 	st.plotly_chart(fig)

		# 	fig2 = go.Figure()
		# 	fig2.add_trace(go.Scatter(x=uk_df.index, y=uk_df['ConfirmedDeaths'],name='United Kingdom Deaths'))
		# 	fig2.add_trace(go.Scatter(x=swe_df.index, y=swe_df['ConfirmedDeaths'],name='Sweden Deaths'))
		# 	fig2.add_trace(go.Scatter(x=sk_df.index, y=sk_df['ConfirmedDeaths'],name='South Korea Deaths'))
		# 	fig2.add_trace(go.Scatter(x=geo_df.index, y=geo_df['ConfirmedDeaths'],name='Georgia Deaths'))
			
		# 	fig2.update_layout(title="Death Count in the 4 Countries")
		# 	st.plotly_chart(fig2)
	elif choice == "Login":
		st.subheader("Login Section")

		username = st.sidebar.text_input("User Name")
		password = st.sidebar.text_input("Password",type='password')
		if st.sidebar.checkbox("Login"):
			# if password == '12345':
			create_usertable()
			hashed_pswd = make_hashes(password)

			result = login_user(username,check_hashes(password,hashed_pswd))
			if result:

				st.success("Logged In as {}".format(username))

				task = st.selectbox("Task",["Add Post","Data","Analytics","Profiles"])
				if task == "Add Post":
					st.subheader("Add Your Post")
				elif task == "Data":
					st.subheader("The link to download the sourcefiles and data is at :")
					st.markdown("1. [Link to the dataset]('https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29')")
					st.markdown("2. [Data Preprocessing source file](https://colab.research.google.com/drive/1C6kOITj4NGa_2l4NDYq8-08B5vKr9Wi7#scrollTo=pNa_XznlaJvV)")
					st.markdown("$\mathcal{Check}$")
					st.markdown("$\pi$ is a number,but $\mu$ is a Statistic.")
					st.markdown("$\mathcal{Everything\quad Counts!}$")
				elif task == "Analytics":
					st.subheader("Analytics")
				elif task == "Profiles":
					st.subheader("User Profiles")
					user_result = view_all_users()
					clean_db = pd.DataFrame(user_result,columns=["Username","Password"])
					st.dataframe(clean_db)
			else:
				st.warning("Incorrect Username/Password")
	#if choice made is to Create a new account/sign-up --->
	elif choice == "SignUp":
		st.subheader("Create New Account")
		new_user = st.text_input("Username")
		new_password = st.text_input("Password",type='password')

		if st.button("Signup"):
			create_usertable()
			add_userdata(new_user,make_hashes(new_password))
			st.success("You have successfully created a valid Account")
			st.info("Go to Login Menu to login")

	#if choice made is to Predict on Bank Member Worthiness --->
	elif choice == 'Prediction':
		st.markdown('For this particular project, you only have to pick 5 variables \
			that are most relevant. These are the Gender, Marital Status,\
			 ApplicantIncome, LoanAmount, and Credit_History ')
		# front end elements of the web page 
		html_temp = """ 
		<div style ="background-color:#87ceeb;padding:13px"> 
		<h1 style ="color:gold;text-align:center;">Streamlit Loan Prediction ML App</h1> 
		</div> 
		"""

		# display the front end aspect
		st.markdown(html_temp, unsafe_allow_html = True) 

		# following lines create boxes in which user can enter data required to make prediction 
		Gender = st.selectbox('Gender',("Male","Female"))
		Married = st.selectbox('Marital Status',("Unmarried","Married")) 
		ApplicantIncome = st.number_input("Applicants monthly income") 
		LoanAmount = st.number_input("Total loan amount")
		Credit_History = st.selectbox('Credit_History',("Unclear Debts","No Unclear Debts"))
		result =""

		# when 'Predict' is clicked, make the prediction and store it 
		if st.button("Predict"): 
			result = prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History) 
			st.success('Your loan is {}'.format(result))
			print(LoanAmount)




if __name__ == '__main__':
	main()