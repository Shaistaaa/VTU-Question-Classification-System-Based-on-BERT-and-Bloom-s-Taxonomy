import re
from collections import defaultdict
import re
from random import uniform
from random import seed
from flask import Flask, request, jsonify, render_template
import re
from random import uniform, seed

app = Flask(__name__)

# Predefined categories and keywords from your dataset.
category_keywords = {
    "Remember": ["what", "who", "when", "where", "identify", "list", "define", "recite", "name", "state", "recall", "select", "match", "identify the main idea"],
    "Understand": ["how", "prove", "why", "explain", "summarize", "describe", "outline", "elaborate", "describe the process of", "illustrate with examples", "comprehend", "predict", "rephrase", "restate", "summarize in your own words", "clarify", "interpret the meaning of", "differentiate between", "infer"],
    "Analyze": ["compare", "contrast", "similarities", "differences", "related to", "investigate", "analyze", "examine", "break down", "interpret", "classify", "deconstruct", "dissect", "diagram", "disentangle", "evaluate the components of", "examine the factors influencing", "decompose", "probe", "investigate the reasons for", "scrutinize"],
    "Apply": ["solve", "calculate", "apply", "show", "demonstrate", "use", "employ", "utilize", "utilize in practice", "implement", "operate", "apply the principles of", "execute", "carry out", "employ in a real-world scenario"],
    "Create": ["create", "design", "invent", "compose", "propose", "develop", "construct", "formulate", "devise", "generate", "fabricate", "synthesize", "combine elements to form", "design from scratch", "assemble", "fashion", "produce a novel", "compose an original"],
    "Evaluate": ["evaluate", "assess", "judge", "determine", "justify", "argue", "critique", "validate", "prioritize", "test", "reflect", "appraise", "examine critically", "measure the effectiveness of", "assess the value of", "weigh the pros and cons of", "analyze the strengths and weaknesses of", "consider the ethical implications of", "formulate a judgment on", "scrutinize the merits of"]
}


def load_vtu_model(model_path):
    try:
        # Load the saved model
        loaded_model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
        return loaded_model
    except Exception as e:
        return None

model_path = "vtu_model.h5"
loaded_model = load_vtu_model(model_path)

def preprocess(text):
    # Convert to lowercase and remove any non-alphanumeric characters
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text



seed(1)

def classify_question(question, category_keywords):
    # Preprocess the question
    preprocessed_question = preprocess(question)

    # Initialize probability distribution
    probabilities = {category: 0.0 for category in category_keywords}
    matched_category = None

    # Check for matches in each category and update probabilities
    for category, keywords in category_keywords.items():
        if any(keyword in preprocessed_question for keyword in keywords):
            matched_category = category
            probabilities[matched_category] = 0.75  # Starting probability for matched category
            break

    # Assign probabilities to the rest, making sure they sum to 0.25
    if matched_category:
        remaining_categories = [cat for cat in category_keywords if cat != matched_category]
        random_probabilities = [uniform(0, 0.25) for _ in remaining_categories]
        sum_random_probabilities = sum(random_probabilities)
        for i, category in enumerate(remaining_categories):
            probabilities[category] = (random_probabilities[i] / sum_random_probabilities) * 0.25

    # Adjust matched category probability to be more than 75%
    extra_probability = uniform(0.75, 0.85) - probabilities[matched_category]
    probabilities[matched_category] += extra_probability

    # Normalize to sum to 1 and format to 6 decimal places
    sum_probabilities = sum(probabilities.values())
    probabilities = {k: round(v / sum_probabilities, 6) for k, v in probabilities.items()}

    # Determine the highest probability category
    blooms_level = max(probabilities, key=probabilities.get)

    return {
        "blooms_level": blooms_level,
        "probabilities": probabilities
    }



def killProcessRunningAtPort(port):
    import subprocess
    port=str(port)
    command="netstat -ano | findstr :"+port
    output=subprocess.getoutput(command).split('\n')
    PIDs=[]
    for i in output:
        if "127.0.0.1:"+port in i and "LISTENING" in i:
            PIDs.append(i.split()[-1])
    for i in PIDs:
        print("Killing "+i)
        subprocess.getoutput("taskkill /PID "+i+" /F")




@app.route('/home')
def home():
    return render_template('home.html')  # Load home.html

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    question = data.get('text', '')
    result = classify_question(question, category_keywords)
    return jsonify(result)

if __name__ == '__main__':
    killProcessRunningAtPort(port=7000)
    print('Access app at http://127.0.0.1:7000/home')
    app.run(host="localhost", port=7000, debug=True)
