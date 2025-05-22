import pandas as pd

german_credit_train = pd.read_csv('german.data', sep=' ', header=None)
german_credit_train.columns = ['checking_account_status', 'duration', 'credit_history', 'purpose', 'credit_amount', 'savings_account', 'employment_status', 'installment_rate', 'personal_status_sex', 'other_debtors', 'present_residence', 'property', 'age', 'other_installment_plans', 'housing', 'number_of_existing_credits', 'job', 'dependents', 'telephone', 'foreign_worker', 'credit_rating']

from sklearn.model_selection import train_test_split

german_credit_train_encoded = german_credit_train.drop(['purpose', 'personal_status_sex'], axis=1)
german_credit_train_encoded = german_credit_train_encoded.drop(['present_residence', 'dependents'], axis=1)
german_credit_train_encoded = pd.get_dummies(german_credit_train_encoded, columns=['telephone'])
german_credit_train_encoded['checking_account_status'], _ = pd.factorize(german_credit_train_encoded['checking_account_status'])
german_credit_train_encoded['credit_history'], _ = pd.factorize(german_credit_train_encoded['credit_history'])
german_credit_train_encoded['savings_account'], _ = pd.factorize(german_credit_train_encoded['savings_account'])
german_credit_train_encoded['employment_status'], _ = pd.factorize(german_credit_train_encoded['employment_status'])
german_credit_train_encoded['other_debtors'], _ = pd.factorize(german_credit_train_encoded['other_debtors'])
german_credit_train_encoded['property'], _ = pd.factorize(german_credit_train_encoded['property'])
german_credit_train_encoded['other_installment_plans'], _ = pd.factorize(german_credit_train_encoded['other_installment_plans'])
german_credit_train_encoded['housing'], _ = pd.factorize(german_credit_train_encoded['housing'])
german_credit_train_encoded['job'], _ = pd.factorize(german_credit_train_encoded['job'])
german_credit_train_encoded['foreign_worker'], _ = pd.factorize(german_credit_train_encoded['foreign_worker'])

X = german_credit_train_encoded.drop(['credit_rating'],axis=1)
y = german_credit_train_encoded['credit_rating'] - 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 38, stratify = y)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # stores the information of the X_train data in the scaler
X_test = scaler.transform(X_test) # just transforms the X_test dataset without computing new mean an SD

X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Save the scaled training features
X_train.to_csv('german_X_train_scaled.csv', index=False)

# Save the training target variable
pd.Series(y_train).to_csv('german_y_train.csv', index=False, header=['credit_rating'])

# Save the scaled testing features
X_test.to_csv('german_X_test_scaled.csv', index=False)

# Save the testing target variable
pd.Series(y_test).to_csv('german_y_test.csv', index=False, header=['credit_rating'])

print("Scaled and split training and testing data saved to CSV files.")