

bd_tokens = ['公司', '主营', '业务', '中国', '境内', '港', '澳', '台', '代理', '销售', '依托', '产品', '研究', '能力', '专业化', '服务', '能力', '团体', '受众', '提供', '投保', '需求', '分析', '投保', '方案', '制定', '保险理赔', '保单', '保全', '一站式', '服务', '最终', '保险公司', '人身保险', '财产', '产品', '代理', '销售', '综合', '服务', '公司', '成立', '中国人民人寿保险股份有限公司', '中德安联人寿保险有限公司', '中英人寿保险有限公司', '阳光人寿保险股份有限公司', '中意人寿保险有限公司', '同方全球人寿保险有限公司', '中国泰康人寿保险股份有限公司', '数十家', '保险公司', '建立', '紧密', '合作关系', '为其', '千余种', '产品', '代理', '销售']
jb_tokens = ['公司', '主营业务', '中国', '境内', '港', '澳', '台', '保险代理', '销售', '依托', '产品', '研究', '能力', '专业化', '服务', '能力', '团体', '个人保险', '受众', '提供', '投保', '需求', '分析', '投保', '方案', '制定', '理赔', '保单', '保全', '一站式', '服务', '最终', '保险公司', '人身保险', '财产保险', '保险产品', '代理', '销售', '综合', '服务', '公司', '成立', '中国', '人寿保险', '股份', '有限公司', '中', '德安', '联', '人寿保险', '有限公司', '中', '英', '人寿保险', '有限公司', '阳光', '人寿保险', '股份', '有限公司', '中意', '人寿保险', '有限公司', '同方', '全球', '人寿保险', '有限公司', '中国', '泰康人寿', '股份', '有限公司', '数十家', '保险公司', '建立', '紧密', '合作', '关系', '千余种', '保险产品', '代理', '销售']
final_tokens = []

bd_length = len(bd_tokens)
jb_length = len(jb_tokens)

bd_idx = 0
jb_idx = 0

pre_bd_str = ''
pre_jb_str = ''

i = 0
while True:

    cur_bd_str = cur_bd_str if pre_bd_str == bd_tokens[bd_idx] else pre_bd_str + bd_tokens[bd_idx]
    cur_jb_str = cur_jb_str if pre_jb_str == jb_tokens[jb_idx] else pre_jb_str + jb_tokens[jb_idx]
    print('{0}:\t Baidu: {1}\t Jieba: {2}\tPreBaidu: {3}\tPreJieba: {4}\tfinal: {5}'.format(str(i), cur_bd_str, cur_jb_str, pre_bd_str, pre_jb_str, str(final_tokens)))

    if cur_bd_str == cur_jb_str:
        if pre_bd_str + pre_jb_str == "":
            final_tokens.append(cur_bd_str)
        pre_jb_str = ""
        pre_bd_str = ""
        bd_idx += 1
        jb_idx += 1
    elif len(cur_bd_str) == len(cur_jb_str):
        if pre_jb_str == "":
            jb_idx += 1
            pre_bd_str = ""
        else:
            pre_jb_str = ""
            bd_idx += 1
    elif len(cur_bd_str) < len(cur_jb_str):
        if cur_bd_str in cur_jb_str:
        # if cur_jb_str.startswith(cur_bd_str):
            bd_idx += 1
            if pre_bd_str != "":
                pre_bd_str = cur_bd_str
                continue
            else:
                pre_bd_str = cur_bd_str
        else:
            pre_bd_str = ""
            pre_jb_str = ""
            bd_idx += 1
            continue
        if cur_jb_str != pre_jb_str:
            final_tokens.append(cur_jb_str)
    elif len(cur_jb_str) < len(cur_bd_str):
        if cur_jb_str in cur_bd_str:
        # if cur_bd_str.startswith(cur_jb_str):
            jb_idx += 1
            if pre_jb_str != "":
                pre_jb_str = cur_jb_str
                continue
            else:
                pre_jb_str = cur_jb_str
        else:
            pre_bd_str = ""
            pre_jb_str = ""
            jb_idx += 1
            continue
        if cur_bd_str != pre_bd_str:
            final_tokens.append(cur_bd_str)


    if bd_idx >= bd_length or jb_idx >= jb_length:
        print("Break. bd_idx={0}, jb_idx={1}".format(str(bd_idx), str(jb_idx)))
        break

    i += 1

print(''.join(bd_tokens))
print(''.join(jb_tokens))
print(''.join(final_tokens))