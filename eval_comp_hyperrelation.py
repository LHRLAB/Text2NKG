import json
import os

def to_gran_format(result_file, label_file, output_file):
    resf = open(result_file, 'r')
    res_dict = json.load(resf)
    testf = open(label_file, "r")
    test_lines = testf.readlines()

    os.remove(output_file)
    rawf = open(output_file, "a")
    res_set = []
    for i in range(0, test_lines.__len__()):
        hypers = []
        test_dict = json.loads(test_lines[i])
        sentence = test_dict["sentences"][0]
        for hyper_relation in res_dict[str(i)]:
            for hr in hyper_relation[1]:
                sub = ""
                obj = ""
                att = ""
                for index in range(hr[0][0], hr[0][1]):
                    sub = sub + sentence[index] + " "
                sub = sub + sentence[hr[0][1]]
                for index in range(hr[1][0], hr[1][1]):
                    obj = obj + sentence[index] + " "
                obj = obj + sentence[hr[1][1]]
                for index in range(hr[3][0], hr[3][1]):
                    att = att + sentence[index] + " "
                att = att + sentence[hr[3][1]]
                hyper = {"N": 3, "relation": hr[2], "subject": sub, "object": obj, hr[4]: [att]}
                hyper = json.dumps(hyper) + "\n"
                hypers.append(hyper)
        rawf.writelines(hypers)
        res_set.append(hypers)
    return res_set


def compaction(res_set, result_comp_file):
    os.remove(result_comp_file)
    resf_comp = open(result_comp_file, "a")
    res_table = []

    for res_line in res_set:
        res_comp_line = []
        #用 map 将主三元组相同的超关系归到一类
        hy_map = {}
        for index in range(res_line.__len__()):
            res_dict = json.loads(res_line[index])
            rso = res_dict["relation"] + res_dict["subject"] + res_dict["object"]
            if rso in hy_map.keys():
                hy_map[rso].append(res_dict)
            else:
                hy_map[rso] = [res_dict]
        # 构建合并后的超关系
        for rso, ds in hy_map.items():
            t_d = {"N": 0}
            for d in ds:
                for k, v in d.items():
                    t_d[k] = v
            t_d["N"] = t_d.__len__() - 2
            res_comp_line.append(json.dumps(t_d))
        res_table.append(res_comp_line)
        formal_res_comp_line = []
        for hyper_relation in res_comp_line:
            formal_res_comp_line.append(hyper_relation + "\n")
        resf_comp.writelines(formal_res_comp_line)
    return res_table


def statistic(res_table, test_file):
    testf = open(test_file, "r")
    test_lines = testf.readlines()
    num_result = 0
    match = 0
    num_label = 0
    N_of_result = {}
    N_of_test = {}
    for i in range(0, test_lines.__len__()):
        res_list = res_table[i]
        test_dict = json.loads(test_lines[i])
        label_relations = test_dict["relations"][0]
        sentence = test_dict["sentences"][0]

        text_label_relations = []
        for label_relation in label_relations:
            sub = ""
            obj = ""
            att = ""
            text_label_relation = {"N": 0}
            for index in range(label_relation[0], label_relation[1]):
                sub = sub + sentence[index] + " "
            sub = sub + sentence[label_relation[1]]
            text_label_relation["relation"] = label_relation[4]
            text_label_relation["subject"] = sub
            for index in range(label_relation[2], label_relation[3]):
                obj = obj + sentence[index] + " "
            obj = obj + sentence[label_relation[3]]
            text_label_relation["object"] = obj

            for att_pair in label_relation[5]:
                for index in range(att_pair[0], att_pair[1]):
                    att = att + sentence[index] + " "
                att = att + sentence[att_pair[1]]
                text_label_relation[att_pair[2]] = [att]
            text_label_relation["N"] = text_label_relation.__len__() - 2
            num_label += 1
            text_label_relations.append(json.dumps(text_label_relation))

        # 在同一个段落里作比较
        for res_hr in res_list:
            num_result += 1
            for label_hr in text_label_relations:
                if res_hr == label_hr:
                    match += 1

        for res_hr in res_list:
            res_hr = json.loads(res_hr)
            if res_hr["N"] in N_of_result.keys():
                N_of_result[res_hr["N"]] += 1
            else:
                N_of_result[res_hr["N"]] = 1

        for label_hr in text_label_relations:
            label_hr = json.loads(label_hr)
            if label_hr["N"] in N_of_test.keys():
                N_of_test[label_hr["N"]] += 1
            else:
                N_of_test[label_hr["N"]] = 1

    print("num_result = " + num_result.__str__())
    print("match = " + match.__str__())
    print("num_label = " + num_label.__str__())
    p = match / num_result if num_result > 0 else 0.0
    r = match / num_label
    f1 = 2 * (p * r) / (p + r) if match > 0 else 0.0
    print("p = " + p.__str__())
    print("r = " + r.__str__())
    print("f1 = " + f1.__str__())
    print(N_of_result)
    print(N_of_test)
    return {"p": p, "r": r, "f1": f1, "N_of_result": N_of_result, "N_of_test": N_of_test}


if __name__ == '__main__':
    # to gran format， result_set 是一个二维list，每一行对应一个段落下抽取的所有超关系
    result_set = to_gran_format(result_file="pred_results(2).json", label_file="test.json", output_file="result.json")
    # 合并 result_set 每一行中的主三元组一样的超关系，
    res_comp_table = compaction(result_set, result_comp_file="result_comp.json")
    # 精确率
    statistic(res_comp_table, test_file="test.json")
