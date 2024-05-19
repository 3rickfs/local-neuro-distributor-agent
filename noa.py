import os
import json
import ctypes
import time
import requests
from datetime import datetime

from flask import Flask, request
from orchestration_planner import (#read_endpoints,
                                   OrchPlannerOps,
                                   read_json_data
                                  )
from synapses import synapses_process
from neuron_distributor import (start_distribution,
                                start_first_layer_input_distribution,
                                send_order_to_nods_to_delete_sp
                               )

app = Flask(__name__)
syn_proc = None

app.config['UPLOAD_FOLDER'] = os.getcwd() + "/uploads"

def delete_sp_obj(syn_proc, synapses_process_id):
    #Delete json file
    os.remove(syn_proc.obj_local_path)

    #Delete S3 obj
    syn_proc.delete_synproc_aimodel_s3_objs()

    #Delete registers
    p = app.config['UPLOAD_FOLDER']
    with open(p + "/synapses_processes.json", "r") as jf:
        sps = json.load(jf)
    jf.close()
    del syn_proc, sps[str(synapses_process_id)]

    #Overriding local registers
    with open(p + "synapses_processes.json", "w") as jf:
        json.dump(sps, jf)
    jf.close()

    #Delete register in remote db
    #noaDBmanager.delete_synproc_register(synapses_process_id)

def get_spcode():
    spcode = str(datetime.now())
    spcode = spcode.replace(" ", "-")
    spcode = spcode.replace(":", "-")
    spcode = spcode.replace("-", "")

    return spcode

def get_synapses_code(spid, uid):
    global syn_proc

    print("Getting synapses code")
    p = app.config["UPLOAD_FOLDER"]
    with open(p + "/synapses_processes.json", "r") as jf:
        synapses_processes = json.load(jf)
    jf.close()

    spc = 0
    pspfn = "persistent_synapses_processes.json"
    with open(p + "/" + pspfn, "r") as jf:
        sps = json.load(jf)
    jf.close()

    try:
        spc = sps[str(spid)]
        fp = os.listdir(p + "/sps")
        files = [
            f for f in fp if int(f.split("-")[0]) == int(uid)
        ]

        for f in files:
            print(f"f: {f}, spc: {spc}")
            if str(spc) in f:
                #then we need to load the sp
                with open(p + "/sps/" + f, "r") as jf:
                    sp = json.load(jf)
                jf.close()

                # reload sp
                syn_proc = synapses_process()
                syn_proc.reload_synaptic_process(sp)
                print(syn_proc)
                fleps = syn_proc.read_fleps()
                print(f"fleps: {fleps}")
                print("lala")

                nspc = id(syn_proc) #new syn proc code
                synapses_processes[str(spid)] = nspc
                sps[str(spid)] = nspc
                spc = nspc
                #update sp json file name 
                sfn = f.split("-")
                sfn[1] = str(nspc)
                nfn = ""
                for i in sfn:
                    nfn += i + "-"
                nfn = nfn[:-1]
                nfp = p + '/sps/' + nfn
                os.rename(p + '/sps/' + f, nfp)
                #Update the local object path
                syn_proc.obj_local_path = nfp

                print(f"New synapses_processes: {synapses_processes}")
                print(f"New persistent sp: {sps}")

                with open(p + "/synapses_processes.json", "w") as jf:
                    json.dump(synapses_processes, jf)
                jf.close()

                with open(p + "/" + pspfn, "w") as jf:
                    json.dump(sps, jf)
                jf.close()
                break
    except Exception as e:
        print(f"Synaptic process error: {e}")
        spc = 0

    return spc

def get_synapses_obj_memory_address(synapses_process_id):
    print("Getting synapses object memory address")
    with open("synapses_processes.json", "r") as jsonfile:
        synapses_processes = json.load(jsonfile)
    jsonfile.close()

    return synapses_processes[str(synapses_process_id)]

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

def send_inputs_to_1layer_nods():
    #TODO: modify this function to send data without a syn_proc obj
    global syn_proc

    input_data = request.get_json()
    user_id = input_data["user_id"]
    username = input_data["username"]
    del input_data["user_id"], input_data["username"]

    sp_c = get_synapses_code(input_data["synapses_process_id"], user_id)
    print(f"sp_c: {sp_c}")
    if sp_c != 0:
        syn_proc = ctypes.cast(
            #int(input_data["synapses_process_id"]),
            int(sp_c),
            ctypes.py_object
        ).value
        print(syn_proc)

        nod_eps = syn_proc.read_fleps()
        #get time to calculate performance
        t = time.time()
        syn_proc.set_pred_start_time(t)
        #distribute inputs to nods
        result = start_first_layer_input_distribution(input_data,
                                                      nod_eps)
        result = {"result": result}
    else:
        result = {"result": "Error - user not available"}

    return json.dumps(result)

def read_synapses_process_output():
    #TODO: modify this function to send data without a syn_proc obj
    global syn_proc

    input_data = request.get_json()
    user_id = input_data["user_id"]
    username = input_data["username"]
    sp_c = get_synapses_code(input_data["synapses_process_id"], user_id)
    print(f"sp id: {input_data['synapses_process_id']}")
    print(f"synaptic code: {sp_c}")
    syn_proc = ctypes.cast(
        #int(input_data["synapses_process_id"]),
        sp_c,
        ctypes.py_object
    ).value
    synapses_output = syn_proc.read_synapses_output()
    res = {
        "synapses_output": synapses_output,
        "pred_time": syn_proc.get_prediction_time(),
        "power_consumption": syn_proc.calculate_pred_power_consumption(),
        "carbon_footprint": syn_proc.calculate_carbon_footprint(),
    }
    #Saving modifications to obj
    syn_proc.export_obj_as_json()

    return res

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
