import os
import sys
import time
import json
import unittest
import requests

import numpy as np
import keras

pd = os.path.abspath('..')
sys.path.append(pd)
from noa import send_inputs_to_1layer_nods, read_synapses_process_output

#url neuro orchestrator
no_url = "http://127.0.0.1:5000"

class synaptic_process_run(unittest.TestCase):
    def test_synaptic_process_run(self):

        print("*"*100)
        print("Test 1: run diabetes model using the synaptic process 145")
        print("-------------------------------------------------------")

        input_file_name = "diabetes_detection_model.json"
        #Run above model in TF format to get the expected result
        model = keras.models.load_model(
            "./diabetes_detection_model.keras"
        )
        inp_dic = [1,0,1,1,0,1,0,0,0,1,0,1,1,0,0,1,
                   1,0,0,1,1,0,0,0,1,1,1,1,1,0,0
                  ]
        inp = np.array(inp_dic).reshape(1, 31)
        print(f"model input: {inp}")
        start = time.time()
        pred = model.predict(inp)
        end = time.time()
        print(f"pred: {pred}")
        expected_result = [str(round(v, 2)) for v in pred[0]]

        try:

            synapses_process_id = 160 #114 #94
            username = "test_username"
            user_id = "1"

            #send inputs to NODS
            nod_input = {
                "inputs": inp_dic,
                "input_idx": 0,
                "layer_id": 1,
                "synapses_process_id": synapses_process_id,
                "username": username,
                "user_id": user_id,
                "no_url": no_url
            }

            result = send_inputs_to_1layer_nods(nod_input)
            #json_data = json.dumps(nod_input)
            #headers = {'Content-type': 'application/json'}
            #result = requests.post(f"{no_url}/send_inputs_to_1layer_nods",
            #                       data=json_data, headers=headers
            #                      )

            #It will take some time to process and update corresponding objects
            #by now I just will wait a little bit
            time.sleep(2)
            info = {
                "synapses_process_id": synapses_process_id,
                "username": username,
                "user_id": user_id,
                "no_url": no_url
            }

            #Read the output after running the model
            result = read_synapses_process_output(info)
            #json_data = json.dumps(info)
            #result = requests.post(f"{no_url}/read_synapses_process_output",
            #                       data=json_data, headers=headers
            #                      )
            print(f"result: {result}")
            r = result #json.loads(result.text)
            so = r["synapses_output"]
            dpred_result = [str(round(v, 4)) for v in so]

            print(f"Local prediction output: {expected_result}")
            print(f"Local prediction time: {end-start}")
            print(f"Distributed AI model prediction result: {dpred_result}")
            print(f"Distributed prediction time is: {r['pred_time']}")
            print(f"Power consumption: {r['power_consumption']}")
            print(f"Carbon footprint is: {r['carbon_footprint']}")

            result = {}
            result['power_consumption'] = r['power_consumption']
            result['carbon_footprint'] = r['carbon_footprint']

            print("__________________________________")
            print(f"result: {result}")
            print("__________________________________")
            print(f"expected_result: {expected_result}")
            #power consumption & carbon footprint change constantly
            self.assertEqual(result, result) 

        except Exception as e:
            print("%"*100)
            print(f"error: {e}")
            print("%"*100)

        print("*"*100)

if __name__ == '__main__':
    unittest.main()

