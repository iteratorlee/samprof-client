package main

import (
	"regexp"
	"strings"

	gpuprofiling "github.com/iteratorlee/samprof-client/go/go-gen"
)

type CriticalNodeType int

const (
	CRITICAL_TYPE_PY_BACKWARD CriticalNodeType = iota
	CRITICAL_TYPE_PY_LOSS
	CRITICAL_TYPE_PY_FORWARD
	CRITICAL_TYPE_TORCH_OP
	CRITICAL_TYPE_TF_OP
	CRITICAL_TYPE_LEAF
	NOT_CRITICAL_NODE
)

func isCriticalNode(node *gpuprofiling.CPUCallingContextNode) CriticalNodeType {
	nodeName := node.GetFuncName()
	if !strings.Contains(nodeName, "python3") {
		if strings.Contains(nodeName, "backward") {
			// fmt.Printf("%v : %v\n", nodeName, "CRITICAL_TYPE_PY_BACKWARD")
			return CRITICAL_TYPE_PY_BACKWARD
		}
		if strings.Contains(nodeName, "loss") {
			// fmt.Printf("%v : %v\n", nodeName, "CRITICAL_TYPE_PY_LOSS")
			//fixme haven't considered pyFileName
			return CRITICAL_TYPE_PY_LOSS
		}
		if strings.Contains(nodeName, "forward") {
			// fmt.Printf("%v : %v\n", nodeName, "CRITICAL_TYPE_PY_FORWARD")
			return CRITICAL_TYPE_PY_FORWARD
		}
	}

	torchOPRegex, _ := regexp.Compile("at::_ops::(\\S+)::call(\\S+)")
	tfOPRegex, _ := regexp.Compile("(\\S+)Op(Kernel)?.+::Compute")
	if match := torchOPRegex.MatchString(nodeName); match {
		// fmt.Printf("%v : %v\n", nodeName, "CRITICAL_TYPE_TORCH_OP")
		return CRITICAL_TYPE_TORCH_OP
	}

	if match := tfOPRegex.MatchString(nodeName); match {
		// fmt.Printf("%v : %v\n", nodeName, "CRITICAL_TYPE_TF_OP")
		return CRITICAL_TYPE_TF_OP
	}

	// fmt.Printf("%v : %v\n", nodeName, "NOT_CRITICAL_NODE")
	return NOT_CRITICAL_NODE
}

func pruneCCT(cpuCCT *gpuprofiling.CPUCallingContextTree) {
	rootId := int64(cpuCCT.GetRootID())
	id2node := cpuCCT.GetNodeMap()

	id2ChildIDs := make(map[int64][]uint64)
	// set parentID
	for nid, node := range id2node {
		// root Node has no parent
		if nid != int64(rootId) {
			for true {
				parentId := int64(node.GetParentID())
				// always keep root node
				if parentId == rootId {
					break
				}
				if parentNode := id2node[parentId]; isCriticalNode(parentNode) != NOT_CRITICAL_NODE {
					break
				} else {
					node.ParentID = parentNode.GetParentID()
				}
			}
		}
		if isCriticalNode(node) != NOT_CRITICAL_NODE {
			if childs, ok := id2ChildIDs[int64(node.GetParentID())]; ok {
				childs = append(childs, uint64(nid))
			} else {
				id2ChildIDs[int64(node.GetParentID())] = []uint64{uint64(nid)}
			}
		}
	}
	for nid, node := range id2node {
		node.ChildIDs = id2ChildIDs[nid]
	}
}
