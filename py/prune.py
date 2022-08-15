import logging
import python_gen as gpuprofiling
import re
from enum import Enum
class CriticalNodeType(Enum):
    CRITICAL_TYPE_PY_BACKWARD=1
    CRITICAL_TYPE_PY_LOSS=2
    CRITICAL_TYPE_PY_FORWARD=3
    CRITICAL_TYPE_TORCH_OP=4
    CRITICAL_TYPE_TF_OP=5
    CRITICAL_TYPE_LEAF=6
    NOT_CRITICAL_NODE=7

def isCriticalNode(node : gpuprofiling.CPUCallingContextNode)->CriticalNodeType:
    nodeName = node.funcName
    if "python3" not in nodeName:
        if 'backward' in nodeName:
            logging.debug(f'{nodeName} is a py backward node')
            return CriticalNodeType.CRITICAL_TYPE_PY_BACKWARD
        elif 'loss' in nodeName:
            logging.debug(f'{nodeName} is a py loss node')
            return CriticalNodeType.CRITICAL_TYPE_PY_LOSS
        elif 'forward' in nodeName:
            logging.debug(f'{nodeName} is a py forward node')
            return CriticalNodeType.CRITICAL_TYPE_PY_FORWARD

    torchOPRegex ="at::_ops::(\\S+)::call(\\S+)"
    tfOPRegex ="(\\S+)Op(Kernel)?.+::Compute"
    if re.match(torchOPRegex, nodeName):
        logging.debug(f'{nodeName} is a torch op')
        return CriticalNodeType.CRITICAL_TYPE_TORCH_OP
    elif re.match(tfOPRegex, nodeName):
        logging.debug(f'{nodeName} is a tf op')
        return CriticalNodeType.CRITICAL_TYPE_TF_OP

    if len(node.childs) == 0:
        logging.debug(f'{nodeName} is a leaf node')
        return CriticalNodeType.CRITICAL_TYPE_LEAF

    logging.debug(f'{nodeName} is not a critical node')
    return CriticalNodeType.NOT_CRITICAL_NODE

def pruneCCT(cpuCCT :gpuprofiling.CPUCallingContextTree):
    rootId = cpuCCT.rootID
    id2node = cpuCCT.nodeMap
    id2childIDs={}
    #skip all non critical nodes
    for nid,node in id2node.items():
        if nid != rootId:
            while True:
                parentId = node.parentID
                if parentId == rootId:
                    break
                parentnode = id2node[parentId]
                if isCriticalNode(parentnode) != CriticalNodeType.NOT_CRITICAL_NODE:
                    break
                else :
                    node.parentID = parentnode.parentID
        if isCriticalNode(node)!=CriticalNodeType.NOT_CRITICAL_NODE:
            parentId = node.parentID
            if parentId not in id2childIDs:
                id2childIDs[parentId]=[]
            id2childIDs[parentId].append(nid)
    #only keep the critical nodes
    for nid,node in id2node.items():
        while len(node.childIDs)>0:
            node.childIDs.pop()
        if nid in id2childIDs:
            for childId in id2childIDs[nid]:
                node.childIDs.append(childId)
