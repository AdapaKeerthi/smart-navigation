from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

X = [
    [0,0,0],
    [1,1,1],
    [3,2,1],
    [6,3,2],
    [10,5,4]
]

y = ["Safe Driver","Safe Driver","Average Driver","Risky Driver","Risky Driver"]

model.fit(X,y)