import requests

phrases = ["Help, my dad is having difficulty breathing",
           "my brother arm is broken",
           "i bumped my head yesterday and have a really bad headache",
           "i am coughing way too much"]

for phrase in phrases:
    json_data = {"data": phrase}

    response = requests.post("http://127.0.0.1:5000/bertriage_api", json=json_data)
    print("Api called")

# import BERTriage.detect
#
# #Loads model
# data = "My head pain"
# nlp_model = BERTriage.detect.load_model(r"D:\ICT Competition\Model\7_class_model.hdf5")
# print(BERTriage.detect.make_prediction(nlp_model,data))
