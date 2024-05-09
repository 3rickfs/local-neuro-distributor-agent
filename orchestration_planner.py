import os
from abc import ABC, abstractmethod
import json

def not_repeat(input_names):
    #neurons mostly have same input names, removing repeatitions  
    #not sure is there is any reason to have different input names.
    #For now just considering one single array for same inputs
    return input_names[0]

def save_nod_info_in_dict(nod_id, capa_id, input_names, pesos, biases, fas,
                          output_names, ops_ep, dis_ep, output_eps, input_num,
                          finns
                         ):
    jsn = {}
    jsn["nod_id"] = str(nod_id)
    jsn["capa_ids"] = capa_id
    jsn["i"] = input_names #not_repeat(input_names)
    jsn["p"] = pesos
    jsn["b"] = biases
    jsn["f"] = fas
    jsn["o"] = output_names
    jsn["ops_ep"] = ops_ep
    jsn["dis_ep"] = dis_ep
    #jsn["output_ep"] = [output_ep[i] for i in range(len(output_ep))]
    jsn["output_eps"] = output_eps
    jsn["input_num"] = input_num
    jsn["finns"] = finns

    return jsn

def get_nods_number(neuron_num, neurons_per_nod):
    div = neuron_num % neurons_per_nod
    if div == 0:
        nods_num = int(neuron_num / neurons_per_nod)
    else:
        nods_num = int(neuron_num / neurons_per_nod) + 1

    return nods_num

def save_files(request, file_path):
    ni, js = '', ''
    nfls = ['json_data', 'nods_info']
    print("is there files?")
    for nf in nfls:
        f = request.files[nf]
        fn = f.filename
        print(f"saving file with name: {f.filename}")
        f.save(os.path.join(file_path, fn))
        if 'json_data' in fn:
            print("json file")
            js = fn
        elif 'nods_info' in fn:
            ni = fn
        else:
            print(f"file {fn} not supported")

    return js, ni

def read_json_data(file_path, fn):
    with open(os.path.join(file_path, fn), 'r') as jf:
        json_data = json.load(jf)
    jf.close()

    return json_data

class orc_pla_ops(ABC):
    """ Orchestration planning operations
    """

    @abstractmethod
    def run_operation(**kwargs):
        #intergace to child clasess
        pass

class get_model_components_v2(orc_pla_ops):
    """ model components with other approach: nods_tech_info
        Former version goes from the neurons to nods, this one
        goes from the other way around, from nods to neurons
    """

    def run_operation(**kwargs):
        json_data = kwargs["json_data"]
        #nod_ep = kwargs["nod_ep"]
        info_layer = []
        # get layers info:
        nc = 1 #neurons counter
        print(len(json_data["layers"]))
        for l in range(1, len(json_data["layers"])+1):
            layer_dict = json_data["layers"]["layer_" + str(l)]
            print("layer_" + str(l))
            neuron_num = len(layer_dict)
            ws, bs, fas, ins, ons, eps = [], [], [], [], [], []
            for n in range(1, neuron_num+1):
                neuron_dict = layer_dict["neuron_" + str(nc)]
                ws.append(neuron_dict["p"]["w"]) #weights
                bs.append(neuron_dict["b"]) #bias
                fas.append(neuron_dict["f"]) #activation function
                ons.append(neuron_dict["o"][0]) #outputs names
                nc += 1
            #inpt names repeats on each neuron as per the json file
            ins = neuron_dict["i"] #inputs names
            #first input name number of each layer
            fonn = int(ons[0][1:])
            finn = int(ins[0][1:])
            print(f"finn: {finn}")
            print(f"fonn: {fonn}")

            info_layer.append([ins, ws, bs, fas, ons, fonn, finn])

        kwargs["info_layer"] = info_layer
        #print(info_layer)

        return kwargs

class get_model_components(orc_pla_ops):
    """ Get model's weights, biases, activation funcitions, etc.
    """
    def run_operation(**kwargs):
        return kwargs

        json_data = kwargs["json_data"]
        nod_ep = kwargs["nod_ep"]
        #split parameters model:
        neurons_per_nod = json_data["CoN_parameters"]["neurons_per_nod"]
        nod_num = json_data["CoN_parameters"]["nod_num"]
        info_layer = []
        # get layers info:
        nc = 1 #neurons counter
        noc = 0 #nod counter
        for l in range(1, len(json_data["layers"])+1):
            layer_dict = json_data["layers"]["layer_" + str(l)]
            neuron_num = len(layer_dict)
            nods_num = get_nods_number(neuron_num, neurons_per_nod)
            ws, bs, fas, ins, ons, eps = [], [], [], [], [], []
            for n in range(1, neuron_num+1):
                neuron_dict = layer_dict["neuron_" + str(nc)]
                ws.append(neuron_dict["pesos"]["w"])
                bs.append(neuron_dict["bias"])
                fas.append(neuron_dict["fa"])
                ins.append(neuron_dict["inputs_names"])
                ons.append(neuron_dict["outputs_names"][0])
                nc += 1
            #list of NOD endpoints in current layer
            eps = nod_ep[noc:noc+nods_num]
            noc += nods_num

            info_layer.append([ins, ws, bs, fas, ons, eps])

        kwargs["info_layer"] = info_layer

        return kwargs

class map_nod_eps(orc_pla_ops):
    """ Nod's endpoints should be mapped to stablish how nods must be considered
        for communicating between themselves and executing neurons according to 
        the layer
    """

    def run_operation(**kwargs):
        print("Mapping nod endpoints")
        nod_ep_map = {}
        nods_tech_info = kwargs["nods_tech_info"]
        neuro_orchestrator_ep = kwargs["neuro_orchestrator_ep"]

        for lo in range(len(kwargs["info_layer"])):
            print(lo)
            nod_ep_map["layer_" + str(lo+1)] = []

        for lo in range(len(kwargs["info_layer"])):
            for nod in nods_tech_info:
                if nods_tech_info[nod]["neuron_dist"][0][0] > 0:
                    for nl in range(len(nods_tech_info[nod]["neuron_dist"])):
                        if nods_tech_info[nod]["neuron_dist"][nl][0] == lo + 1:
                            nid = nods_tech_info[nod]["id"]
                            ep = nods_tech_info[nod]["ops_eps"]
                            nod_ep_map["layer_" + str(nl+1)].append([nid, ep])

        # as noa were the last layer
        lop1 = lo + 2
        nod_ep_map["layer_" + str(lop1)] = []
        nod_ep_map["layer_" + str(lop1)].append([lop1, neuro_orchestrator_ep[0]])
        kwargs["nod_ep_map"] = nod_ep_map
        print(nod_ep_map)

        return kwargs

class create_nod_dictionary_v2(orc_pla_ops):
    """ Distribution of neurons in different layers for each nod
    """
    
    def run_operation(**kwargs):
        print("Creating nod dictionary")
        info_layer = kwargs["info_layer"]
        nods_tech_info = kwargs["nods_tech_info"]
        nods_num = len(nods_tech_info)
        noc = 1
        nec = 0
        nod_dict = {}
        nod_ep_map = kwargs["nod_ep_map"]
        #TODO: nda works for dmodels in which the NOD1'll be for all layers
        #nda = [0 for i in range(len(nods_tech_info["nod_1"]["neuron_dist"]))]
        while True:
            ni = nods_tech_info["nod_" + str(noc)]
            nod_id = noc #for now both are the same
            nod_nd = ni["neuron_dist"]
            distributions = len(nod_nd)
            if nod_nd[0][0] != 0: # accepts neurons to process
                n_i, n_w, n_b, n_f, n_o, n_l, i_n  = [], [], [], [], [], [], []
                n_finn = []
                for d in range(distributions):
                    #if noc == 1:
                    #    nda[d] = nod_nd[d][1]
                    layer = nod_nd[d][0]
                    print(layer)
                    il = info_layer[layer-1]
                    print(f"fonn: {il[5]}")
                    a = nod_nd[d][1] - il[5] #zero position
                    b = nod_nd[d][2] - il[5] + 1 #- nod_nd[d][1] + 1 #it should be > 0
                    n_l.append(layer)
                    n_i.append(il[0])

                    i_n.append(int(il[0][1][1:]) - int(il[0][0][1:]) + 1)   #len(il[0]))

                    n_w.append(il[1][a:b]) #neurons frm a to b
                    n_b.append(il[2][a:b])
                    n_f.append(il[3][a:b])
                    n_o.append(il[4][a:b])
                    n_finn.append(il[6])
                    #print(f"from 0 to {b}, il: {il[1]}")
                i_n.append(len(il[4]))

                #print(n_o)

                #Note: just one nod should contain the last layer for now
                nod_eps = []
                ndl = len(nod_ep_map)
                #for d in range(distributions):
                for nn in range(1, len(nod_ep_map)+1):
                    layer_eps = nod_ep_map["layer_" + str(nn)]
                    for ne in layer_eps:
                        if nod_id == ne[0]:
                            #next layer endpoints
                            lst = nod_ep_map["layer_" + str(nn + 1)]
                            nod_eps.append( #list of ep avoiding self requesting
                                           [e[1] for e in lst if e[0] != nod_id]
                                          )

                nod_dict["nod_" + str(noc)] = save_nod_info_in_dict(
                    noc,
                    n_l,
                    n_i, #info_layer[layer][0],
                    n_w,
                    n_b,
                    n_f,
                    n_o,
                    nods_tech_info["nod_" + str(noc)]["ops_eps"], 
                    nods_tech_info["nod_" + str(noc)]["dis_eps"], 
                    nod_eps,
                    i_n,
                    n_finn
                )
                print(nod_dict["nod_" + str(noc)])

            noc += 1
            if noc >= nods_num:
                break

        kwargs["nod_dict"] = nod_dict

        return kwargs


class create_nod_dictionary(orc_pla_ops):
    """ NOD dictionary in which every NOD can read what it's needed to do
        according to the orchestration plan
    """

    def run_operation(**kwargs):
        return kwargs

        info_layer = kwargs["info_layer"]
        nod_dict = {}
        neuro_orchestrator_ep = kwargs["neuro_orchestrator_ep"]
        neurons_per_nod = kwargs["json_data"]\
                                ["CoN_parameters"]\
                                ["neurons_per_nod"]
        #last layer just one neuron
        layers_num = len(info_layer)
        noc = 1
        for layer in range(layers_num):
            nod_ep_c = 0
            if layer == layers_num - 1: #last layer
                nod_dict["nod_" + str(noc)] = save_nod_info_in_dict(
                    noc,
                    layer + 1,
                    info_layer[layer][0],
                    info_layer[layer][1],
                    info_layer[layer][2],
                    info_layer[layer][3],
                    info_layer[layer][4],
                    info_layer[layer][5][nod_ep_c],
                    neuro_orchestrator_ep #to the noa 
                )
                noc = 1
            else:
                neuron_num = len(info_layer[layer][2])
                for j in range(0, neuron_num, neurons_per_nod):
                    nod_dict["nod_" + str(noc)] = save_nod_info_in_dict(
                        noc,
                        layer + 1,
                        info_layer[layer][0],
                        info_layer[layer][1][j:j+neurons_per_nod],
                        info_layer[layer][2][j:j+neurons_per_nod],
                        info_layer[layer][3][j:j+neurons_per_nod],
                        info_layer[layer][4][j:j+neurons_per_nod],
                        #current nod's ep
                        info_layer[layer][5][nod_ep_c],
                        #ep to the next neuron in next layer
                        info_layer[layer+1][5]
                    )
                    noc += 1
                    nod_ep_c += 1

        kwargs["nod_dict"] = nod_dict

        return kwargs

class OrchPlannerOps:

    @staticmethod
    def run(**kwargs):
        for operation in orc_pla_ops.__subclasses__():
            kwargs = operation.run_operation(**kwargs)

        return kwargs

