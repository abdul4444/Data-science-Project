#Predicting Health Costs. Linear Regression Model

#importing required modules
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

#set path to the csv file
path_dataset = "insurance.csv"

dataset = pd.read_csv(path_dataset)

#converting categorical data to numeric data
df = dataset
label_encoder = preprocessing.LabelEncoder()
df["sex"]= label_encoder.fit_transform(df["sex"])
df["smoker"]= label_encoder.fit_transform(df["smoker"])

label_encoder.fit(df['region'])
region_list = label_encoder.transform(df["region"])
region_mapping = dict(zip(label_encoder.classes_, region_list))

df["region"]= region_list

feature_cols = ["age","sex","bmi","children","smoker","region"]
target = "charges"
#separating feature attributes and target attribute
y = df[target].values
X = df[feature_cols].values

#80/20 hold out approach
#splitting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#training and fitting the model
reg = LinearRegression()  
reg.fit(X_train, y_train)
X_input = [[12,1,28,0,1,2]]
y_pred = reg.predict(X_test)


#plotting actual vs predicted. First n instances
df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
n = 25
df1=df1.head(n)
df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.title('Actual vs Predicted Values')
plt.xlabel('Instance')
plt.ylabel('Charges')
plt.show()

#Printing stats
print("Train R^2 Score: ",reg.score(X_train, y_train))
print("Test R^2 Score: ",reg.score(X_test, y_test))

print("\nExplained Variance Score: ",explained_variance_score(y_test, y_pred))
print("Max Error: ",max_error(y_test, y_pred))
print("Mean Absolute Error: ",mean_absolute_error(y_test, y_pred))
print("Mean Squared Error: ",mean_squared_error(y_test, y_pred))

print("\nIntercept: ",reg.intercept_)
mylist = list(zip(feature_cols, reg.coef_))
print("\nCoefficients:")
for elem in mylist:
        print(elem) 

age = input("Enter age: ")
bmi = input("Enter bmi[i.e 28.7]: ")
gender = input("Enter gender[female=0, male=1]: ")
children = input("Enter children: ")
smoker = input("Enter smoker[yes=1, no=0]: ")
print(region_mapping)
region = input("Enter region number accordingly: ")

#type casting to integer and float
try:
	age = int(age)
	bmi = float(bmi)
	children = int(children)
	region = int(region)
	sex = int(gender)
	smoker = int(smoker)

except ValueError:
	print("\nError: Enter digits only")

#error handling
if(children < 0 ) :
	print("Error: Children must be 0 or more")
	exit()
elif sex < 0 or sex > 1:
	print("Error: Gender must be 0 or 1")
	exit()
elif smoker < 0 or smoker > 1:
	print("Error: smoker must be 0 or 1")
	exit()
elif region < 1 or region > 3:
	print("Error: region must be in range")
	exit()
elif age < 1 :
	print("Error: Age cannot be negative or zero")
	exit()

X_input = [[age,sex,bmi,children,smoker,region]]


y_pred_in=reg.predict(X_input)

print("\nPredicted charges: "+str(y_pred_in[0]))










