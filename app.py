#importing required libraries

from flask import Flask, request, render_template
import pandas as pd

from sklearn.model_selection import train_test_split
from flask import flash
from flask_material import Material
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from sklearn.model_selection import train_test_split


app = Flask(__name__)
Material(app)
app.secret_key="dont tell any one"

@app.route('/')
def home():
    return render_template('login.html')



@app.route('/main')
def main():
    return render_template('index.html')



@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')



@app.route('/',methods=["POST"])
def login():
    if request.method == 'POST':
        username = request.form['id']
        password = request.form['pass']
        if username=='admin' and password=='admin':
            return render_template("index.html")
        else:
            flash("wrong password")
            return render_template("login.html")


data = pd.read_csv("Trainingdata.csv")
data = data.dropna()

# Encoding the class labels
label_encoder = LabelEncoder()
data['CLASS'] = label_encoder.fit_transform(data['CLASS'])

# Tokenizing the sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['SEQ'])
sequences = tokenizer.texts_to_sequences(data['SEQ'])

# Padding the sequences to ensure uniform input size
max_sequence_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, data['CLASS'], test_size=0.2, random_state=42)

# Train the Decision Tree model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Doctor recommendations with image URLs
doctors = {
    'SARS-COV-1': [
        {"name": "Dr. Siddhant Dubey","specialist": "General Physician", "address": "Clinikk Health Hub, Bannerghatta rd, Bengaluru", "qualification": "MBBS", "image": "images/doctors/dr-sidhant.jpg"},
        {"name": "Dr. Renu Saraogi", "specialist": "General Physician", "address": "Rishab Clinic, Bellandur", "qualification": "MBBS", "image": "images/doctors/dr-renu.jpg"},
        {"name": "Dr. Leelamohan PVR", "specialist": "Consultation Physician and General Physician", "address": "Padithem Healthcare, HSR Layout", "qualification": "MBBS, MD", "image": "images/doctors/dr-leelamohan.jpg"}
    ],
    'MERS': [
        {"name": "Dr. Divya K S", "specialist": "Infectious Disease Physician", "address": "Appollo Hospitals, near Mantri Square Mall, Bengaluru", "qualification": "MBBS, MD", "image": "images/doctors/dr-divya.jpg"},
        {"name": "Dr. Sandeep S Reddy", "specialist": "Infectious Disease Specialist", "address": "Brindhavvan Areion Hospital, Chamarajpet", "qualification": "MBBS, MD", "image": "images/doctors/dr-sandeep.jpg"},
        {"name": "Dr. Vinay D", "specialist": "Infectious Disease Physician", "address": "Appollo Hospirals, Bannerughatta Rd, Bengaluru", "qualification": "MBBS, MD, FNB", "image": "images/doctors/dr-vinay.jpg"}
    ],
    'SARS-COV-2': [
        {"name": "Dr. Siddhant Dubey", "specialist": "General Physician", "address": "Clinikk Health Hub, Bannerghatta rd, Bengaluru", "qualification": "MBBS", "image": "images/doctors/dr-sidhant.jpg"},
        {"name": "Dr. Renu Saraogi", "specialist": "General Physician", "address": "Rishab Clinic, Bellandur", "qualification": "MBBS", "image": "images/doctors/dr-renu.jpg"},
        {"name": "Dr. Leelamohan PVR", "specialist": "Consultation Physician and General Physician", "address": "Padithem Healthcare, HSR Layout", "qualification": "MBBS, MD", "image": "images/doctors/dr-leelamohan.jpg"}
    ],
    'Ebola': [
        {"name": "Dr. Divya K S", "specialist": "Infectious Disease Physician", "address": "Appollo Hospitals, near Mantri Square Mall, Bengaluru", "qualification": "MBBS, MD", "image": "images/doctors/dr-divya.jpg"},
        {"name": "Dr. Vinay D", "specialist": "Infectious Disease Physician", "address": "Appollo Hospirals, Bannerughatta Rd, Bengaluru", "qualification": "MBBS, MD, FNB", "image": "images/doctors/dr-vinay.jpg"},
        {"name": "Dr. Neha Mishra", "specialist": "Infectious Disease Specialist", "address": "Manipal Hospitals, HAL old Airport Rd, Bengaluru", "qualification": "MD, MBBS", "image": "images/doctors/dr-neha.jpg"}
    ],
    'Dengue': [
        {"name": "Dr. Abdul Rasheed", "specialist": "General Practitoner and General Physician", "address": "RT Nagar", "qualification": "MBBS", "image": "images/doctors/doctor19.jpg"},
        {"name": "Dr. Ashwitha R Nayak", "specialist": "General Physician", "address": "Clinikk Health Hub, Kammana Halli", "qualification": "MBBS", "image": "images/doctors/dr-ashwitha-r-nayak.jpg"},
        {"name": "Dr. Renuka C", "specialist": "General Physician", "address": "Clinikk Health Hub, Koramangala", "qualification": "MBBS", "image": "images/doctors/dr-renuka-c.jpg"}
    ],
    'Influenza': [
        {"name": "Dr. Raja Selvarajan", "specialist": "General Physician", "address": "Kaveri Healthcare, Domlur", "qualification": "MBBS,MD-Internal Medicine", "image": "images/doctors/dr-raja.jpg"},
        {"name": "Dr. Prashant Dinesh", "specialist": "General Physician and General Practitoner", "address": "Dr. Prashant's Clinic, Ulsoor", "qualification": "MBBS", "image": "images/doctors/dr-prashant.jpg"},
        {"name": "Dr. Prathibha", "specialist": "General Physician and General Practitoner", "address": "Sai Krupa Clinic, Kothanur", "qualification": "MBBS", "image": "images/doctors/dr-prathibha.jpg"}
    ]
}

# Function to predict the class of a new DNA sequence
def predict_sequence(sequence):
    # Tokenize the sequence
    tokenized_seq = tokenizer.texts_to_sequences([sequence])
    # Pad the sequence
    padded_seq = pad_sequences(tokenized_seq, maxlen=max_sequence_length, padding='post')
    # Predict the class
    pred_class = dt.predict(padded_seq)
    return pred_class[0]

@app.route('/main', methods=["POST"])
def analyze():
    if request.method == 'POST':
        dna = request.form['dna']
        result = predict_sequence(dna)

        class_name = label_encoder.inverse_transform([result])[0]
        print("class_name",class_name)

        if class_name == 1:
            class_name='SARS-COV-1'
            result_text = "SARS-COV-1, also known as Severe Acute Respiratory Syndrome Coronavirus 1, is the virus responsible for the SARS outbreak in 2002-2003."
            causes = "The virus is believed to have originated from bats and possibly transmitted to humans through civet cats. It spreads through respiratory droplets."
            precautions = "Avoid close contact with infected individuals, practice good hand hygiene, wear masks in crowded places, and follow quarantine guidelines."
            Treatement="Various medicines—including corticosteroids and the antiviral medicine ribavirin—have been used to treat SARS. But no medicine is known to cure the illness. Doctors continue to search for an effective treatment."
        elif class_name == 2:
            class_name='MERS'
            result_text = "MERS, or Middle East Respiratory Syndrome, is a viral respiratory illness caused by the MERS-CoV coronavirus."
            causes = "MERS-CoV is transmitted to humans from dromedary camels and spreads between people through close contact."
            precautions = "Avoid close contact with infected individuals, wash hands frequently, wear masks, avoid contact with camels, and follow travel advisories."
            Treatement="There is no approved treatment specifically for MERS. Most patients with mild disease recover without complications. Patients with the milder form can be treated at home and take medication for symptoms such as fever and pain. They should stay isolated to avoid spreading the disease. In more severe cases, the patient may develop lung or respiratory failure which requires them to be hospitalized. Doctors may suggest using a breathing tube, a mechanical ventilator or respirator, antibiotics and intravenous fluids."
 
        elif class_name == 3:
            class_name='SARS-COV-2'        
            result_text = "SARS-COV-2, or Severe Acute Respiratory Syndrome Coronavirus 2, is the virus responsible for the COVID-19 pandemic."
            causes = "The virus likely originated from bats and may have been transmitted to humans through an intermediate host. It spreads primarily through respiratory droplets."
            precautions = "Practice social distancing, wear masks, wash hands frequently, avoid large gatherings, and follow vaccination guidelines."
            Treatement="Antibiotics do not kill the SARS-CoV-2 virus, the virus causes COVID-19. Antibiotics are used to treat bacterial infections. Azithromycin and other antibiotics are not recommended to treat COVID-19, unless they are being prescribed an antibiotic for a bacterial infection the patient currently has as well as COVID-19."
 

        elif class_name ==4:
            class_name='Ebola' 
            result_text = "Ebola virus causes Ebola Virus Disease (EVD), a severe and often fatal illness in humans."
            causes = "Ebola is transmitted to people from wild animals and spreads through human-to-human transmission via direct contact with blood, secretions, organs, or other bodily fluids of infected people."
            precautions = "Avoid contact with blood and bodily fluids of infected individuals, use personal protective equipment (PPE), practice safe burial practices, and follow quarantine measures."
            Treatement="Medications. This includes medicines such as monoclonal antibodies that try to stop the virus from reproducing in the body. There are currently two monoclonal antibodies that are FDA approved to treat the Ebola Zaire strain, but more are being developed for other strains of the virus, including the Ebola Sudan strain."
 
        elif class_name == 5:
            class_name='Dengue' 
            result_text = "Dengue virus causes Dengue Fever, a mosquito-borne tropical disease. Symptoms include high fever, headache, vomiting, muscle and joint pains, and a characteristic skin rash."
            causes = "Dengue is transmitted by the bite of Aedes mosquitoes, particularly Aedes aegypti. The mosquitoes become infected when they bite a person already infected with the virus."
            precautions = "Use mosquito repellent, wear long-sleeved clothing, use mosquito nets, eliminate standing water around living areas, and stay in well-screened or air-conditioned areas."
            Treatement="There is no specific treatment for dengue. The focus is on treating pain symptoms. Most cases of dengue fever can be treated at home with pain medicine. Acetaminophen (paracetamol) is often used to control pain."
 

        else:
            class_name = "Influenza"
            result_text = "Influenza, commonly known as the flu, is an infectious disease caused by influenza viruses."
            causes = "Influenza viruses spread mainly through droplets made when people with flu cough, sneeze, or talk. It can also spread by touching a surface or object that has flu virus on it."
            precautions = "Get vaccinated annually, practice good hand hygiene, avoid close contact with infected individuals, cover mouth and nose when coughing or sneezing, and stay home when feeling unwell."
            Treatement="you have a severe infection or are at higher risk of complications, your healthcare professional may prescribe an antiviral medicine to treat the flu. These medicines can include oseltamivir (Tamiflu), baloxavir (Xofluza) and zanamivir (Relenza). Oseltamivir and baloxavir are taken by mouth."
 
        recommended_doctors = doctors[class_name]

        return render_template('contact.html', Treatement=Treatement,class1=class_name, result=result_text, causes=causes, precautions=precautions, res=1, doctors=recommended_doctors)

if __name__ == "__main__":
    app.run(debug=True)

