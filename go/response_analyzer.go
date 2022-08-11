package main

import (
	"fmt"
	"os/exec"
	"sort"
	"strconv"
	"strings"

	gpuprofiling "github.com/iteratorlee/samprof-client/go/go-gen"
)

func getStallReasonDistribution(r *gpuprofiling.GPUProfilingResponse) map[string][]int {
	dist := make(map[string][]int)
	// calculate the distribution of stall reasons
	pcSamplingData := r.GetPcSamplingData()
	for _, pcSampData := range pcSamplingData {
		for _, pcData := range pcSampData.GetPPcData() {
			var stallReasonCnt [38]int
			stallReasons := pcData.GetStallReason()
			for _, stallReason := range stallReasons {
				reasonIdx := stallReason.GetPcSamplingStallReasonIndex()
				stallReasonCnt[reasonIdx] += int(stallReason.GetSamples())
			}
			funcName := pcData.GetFunctionName()
			if _, ok := dist[funcName]; !ok {
				dist[funcName] = make([]int, 38)
			}
			for idx, cnt := range stallReasonCnt {
				dist[funcName][idx] += cnt
			}
		}
	}
	return dist
}

func getCUDAKernelDistribution(r *gpuprofiling.GPUProfilingResponse) map[string]int {
	// calculate the distribution of CUDA kernels
	dist := make(map[string]int)

	pcSamplingData := r.GetPcSamplingData()
	for _, pcSampData := range pcSamplingData {
		for _, pcData := range pcSampData.GetPPcData() {
			var sampleCount int = 0
			for _, stallReason := range pcData.GetStallReason() {
				sampleCount += int(stallReason.GetSamples())
			}
			funcName := pcData.GetFunctionName()
			if _, ok := dist[funcName]; ok {
				dist[funcName] += sampleCount
			} else {
				dist[funcName] = sampleCount
			}
		}
	}

	return dist
}

// get the names of the 3 most frequently occurring kernels in the given distribution
func getTop3KernelNameFromDistribution(kernelDistMap map[string]int) []string {
	nameList := []string{}

	// Import the data in the map into an array
	type KernelCnt struct {
		Name string
		Cnt  int
	}
	var lstKernelCnt []KernelCnt
	for name, cnt := range kernelDistMap {
		lstKernelCnt = append(lstKernelCnt, KernelCnt{name, cnt})
	}

	//Sort in descending order
	sort.Slice(lstKernelCnt, func(i, j int) bool {
		return lstKernelCnt[i].Cnt > lstKernelCnt[j].Cnt
	})

	for i := 0; i < 3 && i < len(lstKernelCnt); i++ {
		nameList = append(nameList, lstKernelCnt[i].Name)
	}

	return nameList
}

// Get the names and their proportions of the top 3 kernels from the kernel distribution
func getTop3KernelFromDistribution(kernelDistMap map[string]int) map[string]float64 {
	var totalCnt int = 0
	for _, cnt := range kernelDistMap {
		totalCnt += cnt
	}

	// Calculate the proportion of the top3 kernels and return
	kernelName2ratio := make(map[string]float64)
	topKernelName := getTop3KernelNameFromDistribution(kernelDistMap)
	for _, kname := range topKernelName {
		kernelName2ratio[kname] = float64(kernelDistMap[kname]) / float64(totalCnt)
	}
	return kernelName2ratio
}

// Get top3 Kernels' name and ratio, from the profiling response
func getTop3Kernel(r *gpuprofiling.GPUProfilingResponse) map[string]float64 {
	kernelDistMap := getCUDAKernelDistribution(r)
	return getTop3KernelFromDistribution(kernelDistMap)
}

func isValidOPName(s string) bool {
	return strings.Contains(s, "ops")
	//return len(s) > 0
	//return strings.Contains(s, "op")
}

// return the root's sample count and record it in the dist
func dfsCCTSampleCount(rootId int64, id2node map[int64]*gpuprofiling.CPUCallingContextNode, id2cnt map[int64]int) int {
	var sampleCount int = 0
	root := id2node[rootId]
	if childIds := root.GetChildIDs(); len(childIds) > 0 {
		// non-leaf node
		for _, childId := range childIds {
			sampleCount += dfsCCTSampleCount(int64(childId), id2node, id2cnt)
		}
	} else {
		// leaf node
		sampleCount = id2cnt[rootId]
	}

	// node id is globally unique
	id2cnt[rootId] = sampleCount

	return sampleCount
}

func getNodeId2NameMap(r *gpuprofiling.GPUProfilingResponse) map[int64]string {
	cpuCCTs := r.GetCpuCallingCtxTree()
	nid2name := make(map[int64]string)
	for _, cpuCCT := range cpuCCTs {
		nodeMap := cpuCCT.GetNodeMap()
		for tmpId, tmpNode := range nodeMap {
			nid2name[tmpId] = tmpNode.GetFuncName()
		}
	}
	return nid2name
}

// Get the map from OP Name to its ratio, from profiling response
func getOPDistribution(r *gpuprofiling.GPUProfilingResponse) map[int64]int {
	// calculate the distribution of OPs

	dist := make(map[int64]int)
	// sample count of leaf nodes
	nid2cnt := make(map[int64]int)
	pcSamplingData := r.GetPcSamplingData()
	for _, pcSampData := range pcSamplingData {
		for _, pcData := range pcSampData.GetPPcData() {
			var sampleCount int = 0
			for _, stallReason := range pcData.GetStallReason() {
				sampleCount += int(stallReason.GetSamples())
			}

			parentId := pcData.GetParentCPUPCID()
			if _, ok := nid2cnt[parentId]; ok {
				nid2cnt[parentId] += sampleCount
			} else {
				nid2cnt[parentId] = sampleCount
			}
		}
	}

	cpuCCTs := r.GetCpuCallingCtxTree()
	//Count the number of samples according to nid, in case some ops appear on multiple links
	for _, cpuCCT := range cpuCCTs {
		rootId := int64(cpuCCT.GetRootID())
		nodeMap := cpuCCT.GetNodeMap()
		dfsCCTSampleCount(rootId, nodeMap, nid2cnt)
	}

	nid2name := getNodeId2NameMap(r)

	for tmpId, tmpName := range nid2name {
		if !isValidOPName(tmpName) {
			continue
		}
		if _, ok := dist[tmpId]; ok {
			dist[tmpId] += nid2cnt[tmpId]
		} else {
			dist[tmpId] = nid2cnt[tmpId]
		}
	}
	return dist
}

// returns Top 3 OPs' ID and ratio
func getTop3OP(r *gpuprofiling.GPUProfilingResponse) map[int64]float64 {
	opDistMap := getOPDistribution(r)
	type OpCnt struct {
		OpId int64
		Cnt  int
	}

	var lstOpCnt []OpCnt
	var totalCnt int = 0
	for k, v := range opDistMap {
		lstOpCnt = append(lstOpCnt, OpCnt{k, v})
		totalCnt += v
	}

	sort.Slice(lstOpCnt, func(i, j int) bool {
		return lstOpCnt[i].Cnt > lstOpCnt[j].Cnt
	})

	nid2ratio := make(map[int64]float64)

	for i := 0; i < 3; i++ {
		opcnt := lstOpCnt[i]
		ratio := float64(opcnt.Cnt) / float64(totalCnt)
		nid2ratio[opcnt.OpId] = ratio
	}

	return nid2ratio
}

// traverse the CCTs and add descendants of OPs to successors
func dfsOPSuccessors(rootId int64, id2node map[int64]*gpuprofiling.CPUCallingContextNode, successors map[int64]float64) {
	root := id2node[rootId]
	if childIds := root.GetChildIDs(); len(childIds) > 0 {
		_, isSuccessor := successors[rootId]
		for _, childId := range childIds {
			if isSuccessor {
				successors[int64(childId)] = 1
			}
			dfsOPSuccessors(int64(childId), id2node, successors)
		}
	}
}

func getTopKernelsOfOP(r *gpuprofiling.GPUProfilingResponse, opId int64) map[string]float64 {
	successors := make(map[int64]float64)
	successors[opId] = 1

	cpuCCTs := r.GetCpuCallingCtxTree()
	for _, cpuCCT := range cpuCCTs {
		nodeMap := cpuCCT.GetNodeMap()
		rootId := int64(cpuCCT.GetRootID())
		dfsOPSuccessors(rootId, nodeMap, successors)
	}

	dist := make(map[string]int)

	// Count the number of times these nodes are sampled
	// save the result in a map from nodeID to sample count
	pcSamplingData := r.GetPcSamplingData()
	for _, pcSampData := range pcSamplingData {
		for _, pcData := range pcSampData.GetPPcData() {
			//Ignore nodes that are not descendants of the specified OP
			parentId := pcData.GetParentCPUPCID()
			if _, ok := successors[parentId]; !ok {
				continue
			}

			var sampleCount int = 0
			for _, stallReason := range pcData.GetStallReason() {
				sampleCount += int(stallReason.GetSamples())
			}

			funcName := pcData.GetFunctionName()
			if _, ok := dist[funcName]; ok {
				dist[funcName] += sampleCount
			} else {
				dist[funcName] = sampleCount
			}
		}
	}

	// Get the first three OPs from the distribution obtained above
	return getTop3KernelFromDistribution(dist)
}

// Get the top three kernels called by the top3 OPs
func getTop3KernelofTop3OP(r *gpuprofiling.GPUProfilingResponse) map[string]float64 {
	//find out the nodes we are concerned about
	successors := getTop3OP(r)
	cpuCCTs := r.GetCpuCallingCtxTree()
	for _, cpuCCT := range cpuCCTs {
		nodeMap := cpuCCT.GetNodeMap()
		rootId := int64(cpuCCT.GetRootID())
		dfsOPSuccessors(rootId, nodeMap, successors)
	}

	dist := make(map[string]int)

	// Count the number of times these nodes are sampled
	// save the result in a map from nodeID to sample count
	pcSamplingData := r.GetPcSamplingData()
	for _, pcSampData := range pcSamplingData {
		for _, pcData := range pcSampData.GetPPcData() {
			//Ignore nodes that are not descendants of the top 3 OPs
			parentId := pcData.GetParentCPUPCID()
			if _, ok := successors[parentId]; !ok {
				continue
			}

			var sampleCount int = 0
			for _, stallReason := range pcData.GetStallReason() {
				sampleCount += int(stallReason.GetSamples())
			}

			funcName := pcData.GetFunctionName()
			if _, ok := dist[funcName]; ok {
				dist[funcName] += sampleCount
			} else {
				dist[funcName] = sampleCount
			}
		}
	}

	// Get the first three OPs from the distribution obtained above
	return getTop3KernelFromDistribution(dist)
}

func getLayerDistribution(r *gpuprofiling.GPUProfilingResponse) map[string]int {
	dist := make(map[string]int)
	// calculate the distribution of Layers
	return dist
}

func toPercentage(x float64) string {
	return strconv.FormatFloat(x*100, 'f', 4, 64) + "%"
}

func printTop3OP(r *gpuprofiling.GPUProfilingResponse) {
	fmt.Println("------------------Top 3 OPs-------------------------------")
	var trimOPName = func(s string) string {
		var leftParen int = 0
		var rightParen int = 0
		for i, c := range s {
			if c == '(' {
				leftParen += 1
			} else if c == ')' {
				rightParen += 1
			}
			if leftParen > 0 && rightParen == leftParen {
				return s[0 : i+1]
			}
		}
		return s
	}
	nid2name := getNodeId2NameMap(r)
	nid2ratio := getTop3OP(r)
	for opid, opratio := range nid2ratio {
		fmt.Printf("%v : %v : %v\n", opid, trimOPName(nid2name[opid]), toPercentage(opratio))
	}
}

func printTop3Kernel(r *gpuprofiling.GPUProfilingResponse) {
	fmt.Println("------------------Top 3 Kernels-------------------------------")
	kernelName2ratio := getTop3Kernel(r)
	for kName, ratio := range kernelName2ratio {
		out, _ := exec.Command("c++filt", kName).Output()
		fmt.Printf("%v : %v\n", string(out), toPercentage(ratio))
	}
}

func printTopKernelsOfTop3OP(r *gpuprofiling.GPUProfilingResponse) {
	top3ops := getTop3OP(r)
	for opId, ratio := range top3ops {
		topKernels := getTopKernelsOfOP(r, opId)
		fmt.Printf("---------------Top Kernels of %v : %v------------\n", opId, toPercentage(ratio))

		for kernelName, kernelRatio := range topKernels {
			out, _ := exec.Command("c++filt", kernelName).Output()
			outputKernelname := strings.Replace(string(out), "\n", "", -1)
			fmt.Printf("%v : %v\n", outputKernelname, toPercentage(kernelRatio))
		}
	}
}
