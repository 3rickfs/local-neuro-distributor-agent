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
from noa import (crear_proceso_sinaptico,
                 getJSONfromKerasModel,
                 delete_proceso_sinaptico)

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
            #Delete sinaptic process
            synapses_process_id = 216
            username = "test_username"
            user_id = "1"

            synaptic_data = {
                "synapses_process_id": synapses_process_id,
                "username": username,
                "user_id": user_id,
                "no_url": no_url
            }

            result = delete_proceso_sinaptico(synaptic_data)
            print(result.text)

            expected_res = "successful"
            self.assertEqual(result.text["result"], expected_res)

        except Exception as e:
            print("%"*100)
            print(f"error: {e}")
            print("%"*100)

        print("*"*100)

if __name__ == '__main__':
    unittest.main()

