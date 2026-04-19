from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(
    [
        [0, 0],
        [1, 1],
        [2, 1],
        [3, 2],
        [4, 3]
    ],
    ["safe", "normal", "careful", "risky", "very risky"]
)