import json
import os

def save_user_feedback(user_input, prediction, file="user_feedback.json"):
    record = {"input": user_input, "prediction": prediction}
    if os.path.exists(file):
        with open(file, "r") as f:
            data = json.load(f)
    else:
        data = []
    data.append(record)
    with open(file, "w") as f:
        json.dump(data, f, indent=2)
