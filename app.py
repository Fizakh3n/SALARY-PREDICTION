from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
pipeline = joblib.load('salary_prediction_pipeline.pkl')  # Ensure pipeline is saved correctly

# Label encoders for user inputs
gender_map = {'Male': 0, 'Female': 1}
education_map = {'Grad': 0, 'Post Grad': 1, 'PhD': 2}
job_map = {'Director of Marketing': 0, 'Director of Operations': 1, 'Others': 2}

@app.route("/", methods=["GET", "POST"])
def home():
    predicted_salary = None
    if request.method == "POST":
        try:
            # Get user inputs
            age = int(request.form["Age"])
            gender = gender_map[request.form["Gender"]]
            education = education_map[request.form["Education Level"]]
            job = job_map[request.form["Job Title"]]
            experience = int(request.form["Years of Experience"])
            
            # Prepare input for prediction
            input_data = [[age, gender, education, job, experience]]
            predicted_salary = pipeline.predict(input_data)[0]
            predicted_salary = round(predicted_salary, 2)
        except Exception as e:
            print(f"Error: {e}")
    
    return render_template("index.html", predicted_salary=predicted_salary)

if __name__ == "__main__":
    app.run(debug=True)
