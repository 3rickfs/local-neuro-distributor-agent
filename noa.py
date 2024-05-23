import os
import json
import ctypes
import time
import requests
from datetime import datetime

from orchestration_planner import (#read_endpoints,
                                   OrchPlannerOps,
                                   read_json_data
                                  )
from synapses import synapses_process
from neuron_distributor import (start_distribution,
                                start_first_layer_input_distribution,
                                send_order_to_nods_to_delete_sp
                               )

syn_proc = None

def delete_sp_obj(syn_proc, synapses_process_id):
    #Delete json file
    #os.remove(syn_proc.obj_local_path)

    #Delete S3 obj
    #syn_proc.delete_synproc_aimodel_s3_objs()

    #Delete registers
    #p = app.config['UPLOAD_FOLDER']
    #with open(p + "/synapses_processes.json", "r") as jf:
    #    sps = json.load(jf)
    #jf.close()
    #del syn_proc, sps[str(synapses_process_id)]

    #Overriding local registers
    #with open(p + "synapses_processes.json", "w") as jf:
    #    json.dump(sps, jf)
    #jf.close()

    #Delete register in remote db
    #noaDBmanager.delete_synproc_register(synapses_process_id)
    print("delete sp obj under dev")


def get_spcode():
    spcode = str(datetime.now())
    spcode = spcode.replace(" ", "-")
    spcode = spcode.replace(":", "-")
    spcode = spcode.replace("-", "")

    return spcode

def get_fleps(nod_info):
    print("Getting first layer endpoints")
    fleps = []
    #print(f"nod info: {nod_info}")
    for nodi in nod_info:
        cid = nod_info[nodi]["capa_ids"][0]
        if cid == 1: #first layer only
            print(f"nod first layer endpoint: {(nod_info[nodi]['ops_ep'])}")
            fleps.append(nod_info[nodi]["ops_ep"])
        else:
            break

    return fleps

def about():
    return "<p>Neuro orchestrator agent"+ \
           " - developed with love by Tekvot dev team. </p>"

def send_inputs_to_1layer_nods(input_data):
    #TODO: the local distributor should sent the inputs too
    no_url = input_data["no_url"]
    del input_data["no_url"]

    json_data = json.dumps(input_data)
    headers = {'Content-type': 'application/json'}
    result = requests.post(
        f"{no_url}/send_inputs_to_1layer_nods",
        data=json_data,
        headers=headers
    )
    result = json.loads(result.text)

    return result

def read_synapses_process_output(input_data):
    no_url = input_data["no_url"]
    del input_data["no_url"]

    json_data = json.dumps(input_data)
    headers = {'Content-type': 'application/json'}
    result = requests.post(
        f"{no_url}/read_synapses_process_output",
        data=json_data,
        headers=headers
    )
    result = json.loads(result.text)

    return result

def crear_proceso_sinaptico(synaptic_data, nods_info):
    global syn_proc

    syn_proc = synapses_process(nods_info)

    synapses_process_id = id(syn_proc)

    # NOA endpoint for getting model output
    noa_url = synaptic_data["neuro_orchestrator_url"]
    neuro_orchestrator_ep = [noa_url + "/set_final_output"]

    #Onboard the model
    data_4_onboarding = {}
    data_4_onboarding["spcode"] = get_spcode()
    data_4_onboarding["neuro_orchestrator_url"] = noa_url
    data_4_onboarding["nods_info"] = nods_info
    data_4_onboarding["noep"] = neuro_orchestrator_ep
    #data_4_onboarding["upload_folder_path"] = file_path
    data_4_onboarding["user_id"] = synaptic_data["user_id"]
    data_4_onboarding["username"] = synaptic_data["username"]
    data_4_onboarding["mj"] = synaptic_data #here comes the model.json too
    data_4_onboarding["sc_fpath"] = "synaptic_process_objs"
    data_4_onboarding["dataset_name"] = synaptic_data["dataset_name"]
    data_4_onboarding["dataset_url"] = synaptic_data["dataset_url"]
    data_4_onboarding["notebook_url"] = synaptic_data["notebook_url"]
    #data for the model
    data_4_onboarding["mfpc"] = "models"
    data_4_onboarding["model_bucket_name"] = "greenbrain"

    try:
        res = syn_proc.onboard_model(**data_4_onboarding)
    except Exception as e:
        print(f"Error when creating synaptic process: {e}")
        res = {"res": f"error - {e}"}

    return res

def delete_proceso_sinaptico():
    json_data = request.get_json()
    synapses_process_id = json_data["synapses_process_id"]
    syn_proc = ctypes.cast(
        int(synapses_process_id),
        ctypes.py_object
    ).value

    try:
        print("Veriying user availability")
        send_order_to_nods_to_delete_sp(syn_proc)
        delete_sp_obj(syn_proc, synapses_process_id)
        res = {"result": "ok"}

    except Exception as e:
        print(f"Error deleting synaptic process obj: {e}")
        res = {"result": "Error: {e}"}

    return res

if __name__ == '__main__':
    app.run(host=host, port=int(port))
