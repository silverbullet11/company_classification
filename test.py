# -*- coding: utf-8 -*-

bd_tokens = ['公司', '主要', '从事', '私募股权投资基金', '管理', '业务', '及', '股权', '投资', '业务', '。', '根据', '中国证监会', '颁布', '的', '《', '上市公司行业分类指引', '》', '（', '2012年', '修订', '）', '，', '公司', '所属', '行业', '为', '“', 'J', '-', '69', '其他', '金融业', '”', '；', '根据', '《', '国民经济行业分类', '》', '（', 'GB T', '4754', '-', '2011', '）', '，', '公司', '所属', '行业', '为', '“', 'J', '-', '69', '其他', '金融业', '”', '。', '根据', '全国中小企业股份转让系统', '《', '挂牌公司管理型行业分类指引', '》', '并', '参考', '同', '行业', '挂牌', '公司', '的', '行业', '分类', '，', '公司', '所属', '行业', '为', '“', '16', '金融', '（', '一级', '行业', '）', '”', '之', '“', '1611证券期货业', '（', '二级', '行业', '）', '”', '之', '“', '161110证券期货业', '（', '三级', '行业', '）', '”', '之', '“', '16111013', '私募基金管理人', '”', '。', '公司', '所', '从事', '的', '私募股权投资基金', '管理', '业务', '，', '是', '指', '公司', '通过', '私募', '的', '方式', '募集', '资金', '、', '设立', '基金', '，', '并', '受托', '管理', '基金', '，', '运用', '基金', '资产', '对', '有', '发展', '潜力', '的', '优质', '企业', '进行', '股权', '投资', '，', '通过', '被', '投资', '企业', '未来', '挂牌', '上市', '、', '并购', '重组', '、', '回购', '或', '其他', '方式', '实现', '退出', '，', '使', '基金', '资产', '实现', '资本', '增值', '和', '投资收益', '，', '为', '基金', '投资', '人', '创造', '收益', '。', '公司', '作为', '基金管理人', '，', '收取', '一定', '的', '基金', '管理', '费用', '和', '基金', '超额', '收益', '报酬', '。', '报告期', '内', '，', '公司', '紧跟', '经济', '转型', '升级', '和', '新兴', '产业', '发展', '趋势', '，', '以', '大', '信息', '、', '大文化', '、', '大', '健康', '产业', '等', '作为', '主要', '投资', '方向', '，', '为', '大量', '移动', '互联网', '等', '新', '商业', '模式', '的', '中小企业', '搭建', '了', '对接', '资本', '市场', '的', '通道', '。', '公司', '股权', '投资', '业务', '主要', '是', '用', '自有资金', '以', '股权', '方式', '投资', '企业', '（', '即', '直接', '投资', '）', '，', '通过', '被', '投资', '企业', '未来', '的', '上市', '等', '获得', '资本', '增值', '、', '投资收益', '。', '报告期', '内', '，', '公司', '股权', '投资', '业务', '占', '比较', '小', '。', '公司', '前期', '的', '私募股权投资基金', '管理', '业务', '及', '直接', '投资', '以', '获取', '资本', '增值', '为', '目的', '的', '财务性', '投资', '为主', '。', '2015年以来', '，', '公司', '积极', '切入', '医疗', '健康', '行业', '的', '产业性', '投资', '（', '即', '战略性投资', '）', '，', '拟', '通过', '资本', '运营', '和', '参与', '企业', '经营', '管理', '，', '利用', '自身', '募集', '资金', '能力', '和', '浙商创投股份有限公司', '公开', '转让', '说明书', '公开', '转让', '说明书', '投资', '管理', '经验', '围绕', '所', '投资', '的', '产业', '平台', '对其', '上下', '游', '行业', '进行', '深度', '整合', '，', '从而', '进一步', '带动', '公司', '的', '私募股权投资基金', '管理', '业务', '和', '直接', '投资', '业务', '。']
jb_tokens = ['公司', '主要', '从事', '私募', '股权', '投资', '基金', '管理', '业务', '及', '股权', '投资', '业务', '。', '根据', '中国证监会', '颁布', '的', '《', '上市公司', '行业', '分类', '指引', '》', '（', '2012', '年', '修订', '）', '，', '公司', '所属', '行业', '为', '“', 'J', '-', '69', '其他', '金融业', '”', '；', '根据', '《', '国民经济', '行业', '分类', '》', '（', 'GB', '/', 'T4754', '-', '2011', '）', '，', '公司', '所属', '行业', '为', '“', 'J', '-', '69', '其他', '金融业', '”', '。', '根据', '全国', '中小企业', '股份', '转让', '系统', '《', '挂牌', '公司', '管理型', '行业', '分类', '指引', '》', '并', '参考', '同行业', '挂牌', '公司', '的', '行业', '分类', '，', '公司', '所属', '行业', '为', '“', '16', '金融', '（', '一级', '行业', '）', '”', '之', '“', '1611', '证券', '期货业', '（', '二级', '行业', '）', '”', '之', '“', '161110', '证券', '期货业', '（', '三级', '行业', '）', '”', '之', '“', '16111013', '私募', '基金', '管理', '人', '”', '。', '公司', '所', '从事', '的', '私募', '股权', '投资', '基金', '管理', '业务', '，', '是', '指', '公司', '通过', '私募', '的', '方式', '募集', '资金', '、', '设立', '基金', '，', '并', '受托', '管理', '基金', '，', '运用', '基金', '资产', '对', '有', '发展潜力', '的', '优质', '企业', '进行', '股权', '投资', '，', '通过', '被', '投资', '企业', '未来', '挂牌', '上市', '、', '并购', '重组', '、', '回购', '或', '其他', '方式', '实现', '退出', '，', '使', '基金', '资产', '实现', '资本', '增值', '和', '投资收益', '，', '为', '基金', '投资人', '创造', '收益', '。', '公司', '作为', '基金', '管理', '人', '，', '收取', '一定', '的', '基金', '管理费用', '和', '基金', '超额', '收益', '报酬', '。', '报告', '期内', '，', '公司', '紧跟', '经济', '转型', '升级', '和', '新兴产业', '发展趋势', '，', '以大', '信息', '、', '大', '文化', '、', '大', '健康', '产业', '等', '作为', '主要', '投资', '方向', '，', '为', '大量', '移动', '互联网', '等', '新', '商业模式', '的', '中小企业', '搭建', '了', '对接', '资本', '市场', '的', '通道', '。', '公司', '股权', '投资', '业务', '主要', '是', '用', '自有', '资金', '以', '股权', '方式', '投资', '企业', '（', '即', '直接', '投资', '）', '，', '通过', '被', '投资', '企业', '未来', '的', '上市', '等', '获得', '资本', '增值', '、', '投资收益', '。', '报告', '期内', '，', '公司', '股权', '投资', '业务', '占', '比较', '小', '。', '公司', '前期', '的', '私募', '股权', '投资', '基金', '管理', '业务', '及', '直接', '投资', '以', '获取', '资本', '增值', '为', '目的', '的', '财务', '性', '投资', '为主', '。', '2015', '年', '以来', '，', '公司', '积极', '切入', '医疗', '健康', '行业', '的', '产业', '性', '投资', '（', '即', '战略性', '投资', '）', '，', '拟', '通过', '资本', '运营', '和', '参与', '企业', '经营', '管理', '，', '利用自身', '募集', '资金', '能力', '和', '浙商', '创投', '股份', '有限公司', '公开', '转让', '说明书', '公开', '转让', '说明书', '投资', '管理', '经验', '围绕', '所', '投资', '的', '产业', '平台', '对', '其', '上下游', '行业', '进行', '深度', '整合', '，', '从而', '进一步', '带动', '公司', '的', '私募', '股权', '投资', '基金', '管理', '业务', '和', '直接', '投资', '业务', '。']

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
            if len(final_tokens) > 0 and cur_jb_str.startswith(final_tokens[-1]):
                final_tokens[-1] = cur_jb_str
            else:
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
            if len(final_tokens) > 0 and cur_bd_str.startswith(final_tokens[-1]) :
                final_tokens[-1] = cur_bd_str
            else:
                final_tokens.append(cur_bd_str)


    if bd_idx >= bd_length or jb_idx >= jb_length:
        print("Break. bd_idx={0}, jb_idx={1}".format(str(bd_idx), str(jb_idx)))
        break

    i += 1

print(''.join(bd_tokens))
print(''.join(jb_tokens))
print(''.join(final_tokens))
