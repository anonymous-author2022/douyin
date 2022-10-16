from ananke.graphs import ADMG
from ananke.identification import OneLineID
from ananke.estimation import CausalEffect
from ananke.datasets import load_afixable_data
from ananke.estimation import AutomatedIF
import numpy as np
import pandas as pd
import networkx as nx


def construct_causal_graph(vertices, di_edges, bi_edges=[]):
    graph = ADMG(vertices, di_edges)
    return graph


def get_ace(G, data, causal, effect, algo="aipw"):
    ace_obj = CausalEffect(graph=G, treatment=causal, outcome=effect)
    ace_ipw = ace_obj.compute_effect(data, algo)
    return ace_ipw


def path_blocked(G, path, z):
    # 对于路径中间的结点逐一判断
    # ['X', 'Z', 'C', 'D', 'Y']
    for i in range(1, len(path) - 1):
        # z -> c <- d
        if G.has_edge(path[i - 1], path[i]) and G.has_edge(path[i + 1], path[i]):
            # 结点path[i]处于对撞结构中
            if path[i] in z:
                # 对撞结点在Z中，此路径无法被此结点阻断，判断下一个结点
                continue

            all_descendants_not_in_z = True
            for des in nx.descendants(G, path[i]):
                # 判断此结点的子孙结点
                if des in z:
                    all_descendants_not_in_z = False
                    break
            if not all_descendants_not_in_z:
                # 某个子孙结点在Z中，此路径无法被此结点阻断，判断下一个结点
                continue

            # 此结点及其子孙结点都不在Z中，此路径可以被此结点阻断，返回True
            print('，且对撞结点', path[i], '被阻断')
            return True

        else:
            # 结点path[i]处于链结构或分叉结构中
            if path[i] in z:
                # 此结点在Z中，此路径可以被此结点阻断，返回True
                print('，且结点', path[i], '被阻断')
                return True
            # 否则此结点无法阻断此路径，继续判断下一个结点

    print('，但未被阻断【×】')
    # 已经判断了此路径中的每一个结点，都无法阻断此路径，返回False
    return False

def d_separation(G, x, y, z):
    all_path_blocked = True
    # 获取图G的一个临时无向图副本，用于搜索简单路径
    U_G = G.to_undirected()
    # 通过U_G获取X和Y之间所有的简单路径
    paths = nx.all_simple_paths(U_G, x, y)
    # 逐一判断每个路径是否被阻断
    for path in paths:
        print('  路径', path, end='')
        blocked = path_blocked(G, path, z)
        if not blocked:
            # 此路径未被阻断，不满足d-分离条件
            all_path_blocked = False

    # 结果输出
    if not all_path_blocked:
        print('有路径未被阻断，不满足d-分离准则')
        return False
    else:
        print('所有路径都已被阻断，满足d-分离准则')
        return True


def frontdoor(G, x, y, z):
    print('正在判断图G中结点集', z, '是否满足关于(', x, ',', y, ')的前门准则')

    # 判断是否为有向无环图
    if not nx.is_directed_acyclic_graph(G):
        print('此图不是有向无环图，无法进行后续判断')
        return False

    # @@@ 判据1：Z切断了所有X到Y的有向路径
    # ------------------------------------------------
    # 获取所有X到Y的有向路径
    directed_paths = nx.all_simple_paths(G, x, y)
    for path in directed_paths:
        print('  路径', path, '是结点', x, '到结点', y, '的有向路径', end='')
        if not path_blocked(G, path, z):
            print('不满足第一条判据，因此不满足前门准则\n')
            return False

    # @@@ 判据2：X到Z没有后门路径（即空集满足(X,Z)的后门准则）
    # ------------------------------------------------
    for node in z:
        backdoor_path_not_exist = backdoor(G, x, node, {})
        if not backdoor_path_not_exist:
            print('不满足第二条判据，因此不满足前门准则\n')
            return False

    # @@@ 判据3：所有Z到Y的后门路径都被X阻断（即X满足(Z,Y)的后门准则）
    # ------------------------------------------------
    for node in z:
        backdoor_path_blocked = backdoor(G, node, y, {x})
        if not backdoor_path_blocked:
            print('不满足第三条判据，因此不满足前门准则\n')
            return False

    # 所有判据皆通过
    print('所有判据通过，满足前门准则\n')
    return True

def backdoor(G, x, y, z):
    print('正在判断图G中结点集', z, '是否满足关于(', x, ',', y, ')的后门准则')

    # 判断是否为有向无环图
    if not nx.is_directed_acyclic_graph(G):
        print('此图不是有向无环图，无法进行后续判断')
        return False

    # @@@ 判据1：Z中没有X的后代结点
    # ------------------------------------------------
    # 获取X的所有后代结点
    des_x = nx.descendants(G, x)
    # 判断Z中是否有X的后代结点
    for node in z:
        if node in des_x:
            print('结点', z, '为', x, '的后代结点，不满足第一条判据，不满足后门准则')
            return False

    # @@@ 判据2：Z阻断了X与Y之间的每条含有指向X的路径
    # ------------------------------------------------
    all_path_blocked = True
    # 获取图G的一个临时无向图副本，用于搜索简单路径
    U_G = G.to_undirected()
    # 通过U_G获取X和Y之间所有的简单路径
    paths = nx.all_simple_paths(U_G, x, y)
    # 逐一判断每个路径是否为指向X的路径，如果是指向X的路径检测其是否被阻断
    for path in paths:
        print('  路径', path, end='')
        if G.has_edge(path[1], path[0]):
            print('是指向X的路径', end='')
            blocked = path_blocked(G, path, z)
            if not blocked:
                # 此路径未被阻断，使得后门准则不满足
                all_path_blocked = False
        else:
            print('不是指向X的路径')

    # 结果输出
    if not all_path_blocked:
        print('有后门路径未被阻断，不满足后门准则\n')
        return False
    else:
        print('所有后门路径都已被阻断，满足后门准则\n')
        return True
