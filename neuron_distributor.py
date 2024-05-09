import requests
import json

def send_order_to_nods_to_delete_sp(sp):
    """ Removing all synaptic process info on nods
    """

    nod_res = []
    spid = sp.synapses_process_id
    eps = [e + "/remove_sp_nod_info" for e in sp.nd_urls]
    for e in eps:
        data = {}
        data["synapses_process_id"] = spid
        json_data = json.dumps(data)
        headers = {'Content-type': 'application/json'}
        print(f"Sending to: {e}")
        result = requests.post(e,
                               data=json_data,
                               headers=headers
                              )
        print(f"result: {result.text}")
        nod_res.append(result.text)

    return nod_res


def start_distribution(nod_dict, synapses_process_id, mfname):
    """ Start distributing neurons to every NOD according
        to the previuosly stablished plan.
    """

    nod_res = []
    for nod_idx in range(1, len(nod_dict) + 1):
        nod_d = nod_dict["nod_" + str(nod_idx)]
        nod_d["synapses_process_id"] = synapses_process_id
        nod_d["mfname"] = mfname 
        #json_data = json.dumps(str(nod_d))
        json_data = json.dumps(nod_d)
        headers = {'Content-type': 'application/json'}
        print(f"Sending to: {nod_d['dis_ep']}")
        result = requests.post(nod_d['dis_ep'],
                               data=json_data,
                               headers=headers
                              )
        print(f"result: {result.text}")
        nod_res.append(result.text)

    return nod_res

def start_first_layer_input_distribution(nod_input, nod_eps):
    """ Distribute same input to every single nod in first layer
    """

    json_data = json.dumps(nod_input)
    nod_res = []
    headers = {'Content-type': 'application/json'}
    for nod_ep in nod_eps:
        print(f"Sending inputs to nod: {nod_ep}")
        result = requests.post(nod_ep,
                               data=json_data,
                               headers=headers
                              )
        nod_res.append(result.text)

    return nod_res
