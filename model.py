from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

# Training data (deviation, stops, confusion)
X = [
    [0,0,0],
    [1,0,0],
    [2,1,0],
    [3,1,1],
    [4,2,1],
    [5,3,2],
    [6,3,3]
]

# Labels
y = [
    "Safe Driver",
    "Safe Driver",
    "Average Driver",
    "Average Driver",
    "Risky Driver",
    "Risky Driver",
    "Dangerous Driver"
]

model.fit(X, y)