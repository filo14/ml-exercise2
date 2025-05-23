import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


titanic_train = pd.read_csv('./train.csv')
original_titanic_train = titanic_train

# drop the Cabin, Ticket, PassengerID, and Name columns as they contain missing values, have no effect, or are unnecessary.
titanic_train = titanic_train.drop(['Cabin','Ticket','PassengerId','Name'], axis=1)

#calculating the mean age by pclass and sex
age_mean_by_pclass_and_sex = titanic_train.groupby(['Pclass', 'Sex'])['Age'].mean()

def set_age_mean(row):
    if pd.isnull(row['Age']):
        return age_mean_by_pclass_and_sex[row['Pclass'], row['Sex']]
    else:

      return row['Age']

titanic_train['Age'] = titanic_train.apply(set_age_mean, axis=1)

#Add missing values for Embarked, since no direct correlation found, inserting most commonly used value
most_common_embarked = titanic_train['Embarked'].mode()[0]
titanic_train['Embarked'] = titanic_train['Embarked'].fillna(most_common_embarked)

# One-hot encoding for Embarked and sex column
titanic_train = pd.get_dummies(titanic_train, columns=['Embarked', 'Sex'])

X = titanic_train.drop(['Survived'],axis=1)
y = titanic_train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 38, stratify = y) # X_train and y_train will be used to train the model

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # stores the information of the X_train data in the scaler
X_test = scaler.transform(X_test) # just transforms the X_test dataset without computing new mean an SD

X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Save the scaled training features
X_train.to_csv('titanic_X_train_scaled.csv', index=False)

# Save the training target variable
pd.Series(y_train).to_csv('titanic_y_train.csv', index=False, header=['Survived'])

# Save the scaled testing features
X_test.to_csv('titanic_X_test_scaled.csv', index=False)

# Save the testing target variable
pd.Series(y_test).to_csv('titanic_y_test.csv', index=False, header=['Survived'])

print("Scaled and split training and testing data saved to CSV files.")