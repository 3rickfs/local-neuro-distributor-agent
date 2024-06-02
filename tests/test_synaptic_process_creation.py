import os
import sys
import time
import json
import unittest
import requests

import numpy as np
import tensorflow as tf

pd = os.path.abspath('..')
sys.path.append(pd)
from noa import crear_proceso_sinaptico, getJSONfromKerasModel

#url neuro orchestrator
no_url = "http://127.0.0.1:5000"
#no_url = "http://a9d4da522d410433185d883825694e22-1486249831.us-west-1.elb.amazonaws.com:5000"
#no_url = "http://a2d0f00821a9540938ba3721318a8679-655184674.us-west-1.elb.amazonaws.com:5000"
#no_url = "http://a4edd8b5e253049c9b077d62a2f081a0-1308811481.us-west-1.elb.amazonaws.com:5000"

class synaptic_process_creations(unittest.TestCase):
    def test_create_synaptic_process(self):
        print("*"*100)
        print("Test 1: Create a synaptic process from a local machine")
        print("-------------------------------------------------------")

        input_model_file_name = "./diabetes_detection_model.json"

        try:
            #Load the AI model JSON
            with open(input_model_file_name, 'r') as jf:
                synaptic_data = json.load(jf)
            jf.close()
            #with open("./nods_info_cloud.json", "r") as jf:
            with open("./nods_info.json", "r") as jf:
                nods_info = json.load(jf)
            jf.close()

            #Distribute neurons
            synaptic_data["user_id"] = 1
            synaptic_data["username"] = "test_username"
            synaptic_data["neuro_orchestrator_url"] = no_url #f"{no_url}/set_final_output"
            synaptic_data["dataset_name"] = "NIST"
            synaptic_data["dataset_url"] = "https://download-nist.com"
            synaptic_data["notebook_url"] = "https://colab.research.google.com/drive/1WE0Pr7r_KAGeaLHtsyeiyBV5LTEJ76q7"

            result = crear_proceso_sinaptico(synaptic_data, nods_info)
            print(result)

            expected_res = "successful"
            self.assertEqual(result["res"], expected_res)

        except Exception as e:
            print("%"*100)
            print(f"error: {e}")
            print("%"*100)


        print("*"*100)

if __name__ == '__main__':
    unittest.main()

