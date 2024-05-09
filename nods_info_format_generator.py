import json

def generate_nods_dis_json(nods_info,
                           neurons_number_per_layer
                          ):
    # Will generate a distribution considering same quantity of neurons per nod


    #neuron_dist: [[1,1,256], [2,513,576], [3,641,650]]
    #neuron_dist: [[1,257,512], [2,577,640]]
    #ndist = [[[1,256], [257,512]], [[513,576], [577,640]], [[641,650]]]
    #[[[1, 256.0], [257.0, 512.0]]]

    ndist = []
    ndc = 0
    nods_num = len(nods_info)
    #print(nods_num)
    layer_num = len(neurons_number_per_layer)
    for l, neurons in enumerate(neurons_number_per_layer):
        layer_pairs = []
        lpc = 0
        for nod in range(1, nods_num + 1):
            pair = []
            if l == 0 and nod == 1: # first layer & nod
                fn = 1
            else:
                #print(f"nod: {nod}")
                if nod != 1: #not first nod no matter what layer
                    #print(layer_pairs)
                    fn = layer_pairs[lpc-1][1] + 1
                elif l != 0 and nod == 1: #not first layer but first nod
                    fn = ndist[ndc-1][-1][1] + 1
            if l == layer_num - 1:
                sn = sum(neurons_number_per_layer)
                pair.append([fn, sn])
                layer_pairs.append(pair[0])
                break
            else:
                if nod == nods_num:
                    sn = sum(neurons_number_per_layer[:l+1])
                else:
                    sn = fn + round(neurons/nods_num) - 1
                pair.append([fn, sn])
                layer_pairs.append(pair[0])
                lpc += 1

        ndist.append(layer_pairs)
        ndc += 1
        #print(ndist)

    nndist = []
    for nod in range(nods_num):
        nod_dist = []
        for j, l in enumerate(ndist):
            #print(nod)
            if nod == 0:
                #print(l[nod])
                nod_dist.append(l[nod])
            else:
                if j < len(ndist) - 1:
                    #print(l[nod])
                    nod_dist.append(l[nod])

        nndist.append(nod_dist)

    nodsinfo = {} 
    for n in range(nods_num):
        nod_info = {
            "id": nods_info["nod_" + str(n+1)]["id"],
            "type": nods_info["nod_" + str(n+1)]["type"],
            "power_consumption": nods_info["nod_" + str(n+1)]["power_consumption"],
            "ram": nods_info["nod_" + str(n+1)]["ram"],
            "arq": nods_info["nod_" + str(n+1)]["arq"],
            "pue": nods_info["nod_" + str(n+1)]["pue"],
            "gco2pkwh": nods_info["nod_" + str(n+1)]["gco2pkwh"],
            "manufacco2": nods_info["nod_" + str(n+1)]["manufacco2"],
            "neuron_dist": nndist[n]
        }

        nodsinfo["nod_" + str(n+1)] = nod_info


    #print(nodsinfo)

    #save json
    with open('./tests/nods_info_exp_1_m.json', 'w') as f:
        json.dump(nodsinfo, f)
    f.close()
    jsonobj = json.dumps(nodsinfo, indent=4)
    print(jsonobj)
    #d = nods_num/neurons

# open json
with open("./tests/nods_info_exp_1.json", "r") as f:
    nods_info = json.load(f)
f.close()

generate_nods_dis_json(nods_info, [512, 128, 10])





